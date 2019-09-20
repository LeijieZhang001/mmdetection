'''
----------------------------------------------------------
  @File:     rpn_det.py
  @Brief:    update from second.pytorch repo
  @Author:   Leijie.Zhang
  @Created:  17:20/9/9/2019
  @Modified: 17:20/9/20/2019
----------------------------------------------------------
'''

import logging
import inspect

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint
from mmdet.core import multi_apply
import cv2

from ..losses import smooth_l1_loss, FocalLoss, SmoothL1Loss
from ..registry import DETECTORS
from .pc_utils import *
from .pc_losses import *

from mmdet.datasets.pc_utils import *
from tools.vis import *

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[1]
        self.nx = output_shape[2]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 output_shape=(64,496,432)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        """

        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        self.output_shape = output_shape

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            logger.info('initialize the feat extractor model without pretrained......')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, feats, coords, **kwargs):
        assert len(feats) == len(coords), 'feats & coords batch size mismatch.'
        for feat, coord in zip(feats, coords):
            assert feat.shape[0] == coord.shape[0], 'feat & coord shape mismatch'
        self.batch_size = len(feats)
        # Forward pass through PFNLayers
        # P x 100 x 9 => P x 64
        out_feats = []
        for feat in feats:
            for pfn in self.pfn_layers:
                feat = pfn(feat)
            out_feats.append(feat)
        feats = out_feats
        assert len(feats) == len(coords), '2 feats & coords batch size mismatch.'
        for feat, coord in zip(feats, coords):
            assert feat.shape[0] == coord.shape[0], '2 feat & coord shape mismatch'

        batch_convas = []
        for feat, coord in zip(feats, coords):
            coord = coord.reshape((-1, coord.shape[-1]))
            feat = feat.reshape((-1, feat.shape[-1]))
            assert feat.shape[0] == coord.shape[0], '3 feat & coord shape mismatch'
            coord = list(coord.transpose(1,0))
            convas = torch.zeros(self.output_shape, dtype=feat.dtype, device=feat.device)
            convas[:,list(coord[0]), list(coord[1])] = feat.transpose(1,0)
            batch_convas.append(convas)

        batch_convas = torch.stack(batch_convas, 0).view((self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))

        return batch_convas

#class RPNNoHeadBase(nn.Module):
class RPNNoHeadBase(PillarFeatureNet):
    def __init__(self,
                 use_norm,
                 num_class,
                 layer_nums,
                 layer_strides,
                 num_filters,
                 upsample_strides,
                 num_upsample_filters,
                 num_input_features,
                 num_anchor_per_loc,
                 encode_background_as_zeros,
                 use_direction_classifier,
                 use_groupnorm,
                 num_groups,
                 box_code_size,
                 num_direction_bins,
                 name,
                 pillars,
                 input_chw=None):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        self._pillars = pillars
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        if input_chw:
            assert input_chw[0] == num_input_features, 'num_input_features should be equal to input_chw[0]'
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, feats, **kwargs):
        x = feats
        if self._pillars:
            x = super().forward(x, **kwargs)
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res

class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 input_chw=None,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=10, #128
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=False,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 pillars=False,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name,
            pillars=pillars,
            input_chw=input_chw)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        self._encode_background_as_zeros = encode_background_as_zeros
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, feats, **kwargs):
        x = feats
        res = super().forward(x, **kwargs)
        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        bbox_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 3, 4, 1, 2).contiguous()
        if self._encode_background_as_zeros:
            cls_scores = cls_preds.view(-1, self._num_anchor_per_loc,
                                    self._num_class, H, W).permute(
                                        0, 1, 3, 4, 2).contiguous()
        else:
            cls_scores = cls_preds.view(-1, self._num_anchor_per_loc,
                                    self._num_class+1, H, W).permute(
                                        0, 3, 4, 1, 2).contiguous()

        # ret_dict = {
        #     "box_preds": box_preds,
        #     "cls_preds": cls_preds,
        # }
        outs = dict(cls_scores=cls_scores,
                    bbox_preds=bbox_preds)
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            #ret_dict["dir_cls_preds"] = dir_cls_preds
            outs.update(dir_cls_preds=dir_cls_preds)

        # cls_scores:    Nx2xHxWx2
        # bbox_preds:    Nx2xHxWx7
        # dir_cls_preds: Nx2xHxWx2
        return outs

@DETECTORS.register_module
class RPNDet(RPNBase):

    def __init__(self, train_cfg=None, test_cfg=None, pretrained=None,
                 input_chw=[10, 496, 432],
                 sizes=[1.6, 3.9, 1.56],
                 strides=[0.4, 0.4, 1.0],
                 offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 class_name=None,
                 matched_threshold=-1,
                 unmatched_threshold=-1,
                 use_sigmoid_score=False,
                 **kwargs):
        super(RPNDet, self).__init__(input_chw, **kwargs)
        self.anchor_generator = AnchorGeneratorStride(sizes=sizes,
                                                      anchor_strides=strides,
                                                      anchor_offsets=offsets,
                                                      rotations=rotations,
                                                      match_threshold=matched_threshold,
                                                      unmatch_threshold=unmatched_threshold,
                                                      class_name=class_name)
        # anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
        # 7: tx, ty, tz, w, l, h, ry
        # typically: 496/2 x 432/2 x 1 x 2 x 7
        self._use_sigmoid_score = use_sigmoid_score
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.input_chw = input_chw
        anchor_map = [i/2 for i in self.input_chw]
        self.anchors = self.anchor_generator.generate(anchor_map)
        self.anchors = self.anchors[0,...]
        print('anchors size: ', self.anchors.shape)
        self.similarity_fn = similarity_fn
        self.box_encoding_fn = GroundBox3dCoder().encode
        self.box_decoding_fn = GroundBox3dCoder().decode

        #self.voxel_feat_extractor = PillarFeatureNet()

        self.init_weights(pretrained)

        self.focalloss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes

    def forward(self, feats, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(feats, **kwargs)
        else:
            return self.forward_test(feats, **kwargs)

    def forward_train(self, feats, **kwargs):
        outs = super().forward(feats, **kwargs)
        if self._use_direction_classifier:
            return self.loss(outs['cls_scores'], outs['bbox_preds'], dir_cls_preds=outs['dir_cls_preds'], feats=feats, cfg=self.train_cfg, **kwargs)
        else:
            return self.loss(outs['cls_scores'], outs['bbox_preds'], cfg=self.train_cfg, **kwargs)

    def forward_test(self, feats, pc_meta, **kwargs):
        outs = super().forward(feats, **kwargs)

        assert outs['cls_scores'].size()[1:3] == self.anchors.shape[:2], 'featmap size to generate anchors should be equal.'
        batch_size = outs['cls_scores'].size()[0]
        all_anchors = [self.anchors.reshape((-1, self.anchors.shape[-1])) for _ in range(batch_size)]
        all_anchors = np.vstack(all_anchors)
        all_anchors = all_anchors.reshape(batch_size, -1, all_anchors.shape[-1])

        batch_cls_scores = outs['cls_scores'].reshape((outs['cls_scores'].size()[0], -1, outs['cls_scores'].size()[4]))
        batch_bbox_preds = outs['bbox_preds'].reshape((outs['bbox_preds'].size()[0], -1, outs['bbox_preds'].size()[4]))
        batch_bbox_preds = torch.tensor(self.box_decoding_fn(batch_bbox_preds.cpu().numpy(), all_anchors)).cuda()

        self._use_direction_classifier = False

        if self._use_direction_classifier:
            batch_dir_preds = outs['dir_cls_preds'].reshape((outs['dir_cls_preds'].size()[0], -1, outs['dir_cls_preds'].size()[4]))
        else:
            batch_dir_preds = [None] * batch_size

        num_class_with_bg = self._num_class
        predictions_dicts = []
        post_center_range = None
        for box_preds, cls_preds, dir_preds, pc_me in zip(
                batch_bbox_preds, batch_cls_scores, batch_dir_preds, pc_meta):
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            self._use_rotate_nms = True
            if self._use_rotate_nms:
                nms_func = rotate_nms
            else:
                nms_func = nms
            feature_map_size_prod = batch_bbox_preds.shape[
                1] // 2 # self.target_assigner.num_anchors_per_location
            if False: #self._multiclass_nms:
                assert self._encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = self._nms_score_thresholds
                pre_max_sizes = self._nms_pre_max_sizes
                post_max_sizes = self._nms_post_max_sizes
                iou_thresholds = self._nms_iou_thresholds
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                        range(self._num_class),
                        score_threshs,
                        pre_max_sizes, post_max_sizes, iou_thresholds):
                    if True: #self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1,
                            self._num_class)[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1,
                                                     boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigner.anchors_range(class_idx)
                        class_scores = total_scores.view(
                            -1,
                            self._num_class)[anchors_range[0]:anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1,
                            boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds.view(-1,
                            box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1])
                        if self._use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.contiguous(
                            ).view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[
                                class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[
                                class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores, pre_ms,
                                        post_ms, iou_th)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full([class_boxes[selected].shape[0]],
                                       class_idx,
                                       dtype=torch.int64,
                                       device=box_preds.device))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(
                                class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                nms_pre_max_sizes = 1000
                nms_post_max_sizes = 300
                nms_score_thresholds = 0.05
                nms_iou_thresholds = 0.1
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(
                        total_scores, dim=-1)
                if nms_score_thresholds > 0.0:
                    top_scores_keep = top_scores >= nms_score_thresholds
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if nms_score_thresholds > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=nms_pre_max_sizes,
                        post_max_size=nms_post_max_sizes,
                        iou_threshold=nms_iou_thresholds,
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    self._dir_limit_offset = 1
                    self._dir_offset = 0
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = torch_limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        self._dir_limit_offset, period)
                    box_preds[
                        ...,
                        6] = dir_rot + self._dir_offset + period * dir_labels.to(
                            box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        'id': pc_me['id'],
                        "box3d_preds": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                    }
                else:
                    predictions_dict = {
                        'id': pc_me['id'],
                        "box3d_preds": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                    }
            else:
                dtype = batch_bbox_preds.dtype
                device = batch_bbox_preds.device
                predictions_dict = {
                    'id': pc_me['id'],
                    "box3d_preds": torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        #self.voxel_feat_extractor.init_weights(pretrained)
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            logger.info('initialize the model without pretrained......')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             pc_meta,
             cfg,
             dir_cls_preds=None,
             feats=None,
             gt_bboxes_ignore=None,
             **kwargs):

        print('cls_scores size: ', cls_scores.size())
        assert len(gt_bboxes) == cls_scores.size()[0], 'batch size should be equal.'
        batch_size = len(gt_bboxes)
        print('batch_size: ', batch_size)
        print('cls_scores size: ', cls_scores.size())
        print('bbox_preds size: ', bbox_preds.size())
        #print('dir_cls_preds size: ', dir_cls_preds.size())
        print('gt_bboxes size: ', gt_bboxes[0].size())
        print('gt_labels: ', gt_labels[0].size())
        assert cls_scores.size()[1:3] == self.anchors.shape[:2], 'featmap size to generate anchors should be equal.'

        cls_scores = cls_scores.reshape((cls_scores.size()[0], -1, cls_scores.size()[4]))
        bbox_preds = bbox_preds.reshape((bbox_preds.size()[0], -1, bbox_preds.size()[4]))
        dir_cls_preds = dir_cls_preds.reshape((dir_cls_preds.size()[0], -1, dir_cls_preds.size()[4]))
        print('--- after reshape')
        print('cls_scores size: ', cls_scores.size())
        print('bbox_preds size: ', bbox_preds.size())
        #print('dir_cls_preds size: ', dir_cls_preds.size())

        all_anchors = [self.anchors.reshape((-1, self.anchors.shape[-1])) for _ in range(batch_size)]
        #from functools import partial
        #res = multi_apply(get_anchor_target,
        #                  all_anchors,
        #                  gt_bboxes,
        #                  self.similarity_fn,
        #                  self.box_encoding_fn)
        labels = []
        bbox_targets = []
        bbox_outside_weights = []
        importance = []
        for i in range(batch_size):
            res = get_anchor_target(self.anchors.reshape((-1, self.anchors.shape[-1])),
                                    gt_bboxes[i].cpu().numpy(),
                                    self.similarity_fn,
                                    self.box_encoding_fn,
                                    neg_pos=20,
                                    rpn_batch_size=512)
            labels.append(torch.Tensor(res['labels']).type(torch.int32))
            bbox_targets.append(torch.Tensor(res['bbox_targets']))
            bbox_outside_weights.append(torch.Tensor(res['bbox_outside_weights']))
            importance.append(torch.Tensor(res['importance']))

        # labels:       batch x anchors
        # cls_scores:   batch x anchors x 2
        # bbox_targets: batch x anchors x 7
        labels = torch.stack(labels).type(torch.long).cuda()
        bbox_targets = torch.stack(bbox_targets).cuda()

        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)
        targets = labels #.reshape(-1)
        targets[targets<0] = 0
        print('targets max: ', targets.max())
        loss_cls_all = F.cross_entropy(cls_scores.reshape(-1, cls_scores.shape[-1]), targets.reshape(-1), reduction='none')
        #loss_cls_all = self.focalloss(cls_scores.reshape(-1, cls_scores.shape[-1]), targets, reduction_override='none')
        print('loss_cls_all size: ', loss_cls_all.shape)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = neg_inds.size(0)
        print('num_pos_samples: ', num_pos_samples)
        print('num_neg_samples: ', num_neg_samples)
        #assert num_pos_samples*100 >= num_neg_samples, 'pos neg ration error.'
        loss_cls_pos = loss_cls_all[pos_inds].sum() / num_pos_samples / batch_size
        loss_cls_neg = loss_cls_all[neg_inds].sum() / num_pos_samples / batch_size

        loc_loss_weight = 2.0
        loss_loc_all = self.smooth_l1_loss(bbox_preds, bbox_targets, reduction_override='none')
        loss_loc = loc_loss_weight * loss_loc_all.reshape(-1, bbox_preds.shape[-1])[pos_inds,:].sum() / num_pos_samples / batch_size


        total_scores = torch.sigmoid(cls_scores.reshape(-1, cls_scores.shape[-1]))
        #total_scores = F.softmax(cls_scores.reshape(-1, cls_scores.shape[-1]), dim=-1)
        pos_scores = total_scores[pos_inds,:]
        print('pos_scores: ', pos_scores)
        neg_scores = total_scores[neg_inds,:]
        print('neg_scores: ', neg_scores)

        #print('feats size: ', feats.shape)
        #fea = feats.cpu().numpy()
        labels = labels.cpu().numpy().reshape(batch_size,-1)
        anchors = self.anchors.reshape((-1, self.anchors.shape[-1]))
        total_scores = total_scores.cpu().detach().numpy().reshape(batch_size, -1, 2)
        #for i in range(fea.shape[0]):
            #bv = debug_feats(fea[i,...], channel=5, show=False)
        for i in range(len(feats)):
            bv = np.zeros((496, 432, 3), dtype = np.uint8)

            ## pos anchors
            pos_inds = (total_scores[i,:,1].reshape(-1) > 0.5)
            pos_anc = anchors[pos_inds,:].reshape((-1, anchors.shape[-1]))
            bv = plot_bbox3d(bv, pos_anc, scalar=(25,188,0))

            ## assigned anchors
            pos_inds = (labels[i,...] > 0).nonzero()
            neg_inds = (labels[i,...] == 0).nonzero()
            anc = anchors[pos_inds,:].reshape((-1, anchors.shape[-1]))
            bv = plot_bbox3d(bv, anc, scalar=(125,92,0))

            # gt_boxes
            boxes_3d = gt_bboxes[i].cpu().numpy()
            bv = plot_bbox3d(bv, boxes_3d)

            cv2.imshow('hh', bv)
            cv2.waitKey(2)



        # #print('labels.size: ', labels[0].shape)
        # #print('bbox_targets.size: ', bbox_targets[0].shape)
        # #print('bbox_outside_weights.size: ', bbox_outside_weights[0].shape)
        # labels = torch.stack(labels).type(torch.int32).cuda()
        # bbox_targets = torch.stack(bbox_targets).cuda()
        # bbox_outside_weights = torch.stack(bbox_outside_weights).cuda()
        # importance = torch.stack(importance).cuda()
        # print('labels.size: ', labels.size())
        # print('labels > 0 size', labels[labels>0].shape)
        # print('labels > 0', labels[labels>0])
        # print('bbox_targets.size: ', bbox_targets.size())
        # print('bbox_outside_weights.size: ', bbox_outside_weights.size())
        # print('importance.size: ', importance.size())

        # print('+++++++++ end ++++++++++')

        # cls_weights, reg_weights, cared = prepare_loss_weights(labels, dtype=bbox_preds.dtype)
        # cls_targets = labels * cared.type_as(labels)
        # cls_targets = cls_targets.unsqueeze(-1)

        # loc_loss, cls_loss = create_loss(
        #     box_preds=bbox_preds,
        #     cls_preds=cls_scores,
        #     cls_targets=cls_targets,
        #     cls_weights=cls_weights * importance,
        #     reg_targets=bbox_targets,
        #     reg_weights=reg_weights * importance,
        #     num_class=self._num_class,
        #     encode_rad_error_by_sin=True,
        #     encode_background_as_zeros=self._encode_background_as_zeros,
        #     box_code_size=7,
        #     sin_error_factor=1.0)

        # loc_loss_reduced = loc_loss.sum() / batch_size
        # loc_loss_reduced *= 2.0 # self._loc_loss_weight
        # cls_pos_loss, cls_neg_loss = get_pos_neg_loss(cls_loss, labels)
        # cls_pos_loss /= 1.0 # self._pos_cls_weight
        # cls_neg_loss /= 1.0 # self._neg_cls_weight
        # cls_loss_reduced = cls_loss.sum() / batch_size
        # cls_loss_reduced *= 1.0 # self._cls_loss_weight
        # loss = loc_loss_reduced + cls_loss_reduced
        # print('cls_loss > 0 size: ', cls_loss[cls_loss>0].shape)
        # print('cls_loss > 0: ', cls_loss[cls_loss>0])
        # print('loc_loss > 0 size: ', loc_loss[loc_loss>0].shape)
        # print('loc_loss > 0: ', loc_loss[loc_loss>0])
        # print('cls_loss_reduced: ', cls_loss_reduced)
        # print('cls_pos_loss: ', cls_pos_loss)
        # print('cls_neg_loss: ', cls_neg_loss)

        # if self._use_direction_classifier:
        #     dir_targets = get_direction_target(
        #         torch.Tensor(np.vstack(all_anchors)).cuda(),
        #         bbox_targets,
        #         dir_offset=0,
        #         num_bins=2)
        #     dir_logits = dir_cls_preds.view(
        #         batch_size, -1, 2)
        #     weights = (labels > 0).type_as(dir_logits) * importance
        #     weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
        #     dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        #     dir_loss = dir_loss_ftor(
        #         dir_logits, dir_targets, weights=weights)
        #     dir_loss = dir_loss.sum() / batch_size
        #     loss += dir_loss * 0.2

        #return dict(loss_cls=cls_loss_reduced, loss_bbox=loc_loss_reduced, loss_dir=dir_loss, loss=loss, loss_cls_pos=cls_pos_loss, loss_cls_neg=cls_neg_loss)
        #return dict(loss_cls_pos=loss_cls_pos, loss_cls_neg=loss_cls_neg)
        return dict(loss_cls_pos=loss_cls_pos, loss_cls_neg=loss_cls_neg, loss_loc=loss_loc)

        # cls_scores:    batch_size * anchors * 2
        # bbox_preds:    batch_size * anchors * 7
        # labels:        batch_size * anchors
        # label_weights: batch_size * anchors
        # bbox_targets:  batch_size * anchors * 7
        # bbox_weights:  batch_size * anchors

        # cls_scores = cls_scores.reshape(-1, cls_scores.size()[-1])
        # labels = labels.reshape(-1).type(torch.int64)
        # bbox_outside_weights = bbox_outside_weights.reshape(-1)
        # print('begin to cross entropy')
        # print('cls_scores: ', cls_scores.size())
        # print('labels: ', labels.size())
        # loss_cls = F.cross_entropy(cls_scores, labels, reduction='none') # * bbox_outside_weights
        # print('end cross entropy')
        # #pos_inds = (labels > 0).nonzero().view(-1)
        # #neg_inds = (labels == 0).nonzero().view(-1)

        # print('loss_cls.size: ', loss_cls.size())

        # #num_pos_samples = pos_inds.size(0)
        # #num_neg_samples = 3 * num_pos_samples
        # #if num_neg_samples > neg_inds.size(0):
        # #    num_neg_samples = neg_inds.size(0)
        # #topk_loss_cls_neg, _ = loss_cls[neg_inds].topk(num_neg_samples)
        # #loss_cls_neg = topk_loss_cls_neg.sum()
        # #loss_cls_pos = loss_cls[pos_inds].sum()
        # #num_total_samples = num_pos_samples + num_neg_samples
        # #loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        # print('loss_cls: ', loss_cls)

        # loss_bbox = smooth_l1_loss(
        #     bbox_preds,
        #     bbox_targets,
        #     bbox_outside_weights,
        #     1,               # beta=cfg.smoothl1_beta,
        #     avg_factor=num_total_samples)
        # #print("loss_bbox: ", loss_bbox.size())
        # #return loss_cls[None], loss_bbox


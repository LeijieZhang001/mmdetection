# model settings
model = dict(
    type='RPNDet',
    #pretrained='/home/leijie/coding/second.pytorch/work_dir/xyres_16.pt.bk/voxelnet-296960.tckpt',
    pillars=True,
    num_class=1,
    input_chw=(64, 496, 432),
    num_input_features=64,
    layer_nums=(3, 5, 5),
    layer_strides=(2, 2, 2),
    num_filters=(64, 128, 256),
    upsample_strides=(1, 2, 4),
    num_upsample_filters=(128, 128, 128),
    use_groupnorm=False,
    num_groups=32,
    use_direction_classifier=True,
    # anchor_generator
    sizes=(1.6, 3.9, 1.56), # wlh
    strides=(0.32, 0.32, 0.0), # if generate only 1 z_center, z_stride will be ignored
    offsets=(0.16, -39.52, -1.78), # origin_offset + strides / 2
    rotations=(0, 1.57), # 0, pi/2
    matched_threshold=0.6,
    unmatched_threshold=0.45,
    class_name="Car")
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        use_rotate_nms=False,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_score_threshold=0.05,
        nms_iou_threshold=0.5,

        sample_positive_fraction=-1,
        sample_size=512,
        assign_per_class=True,

        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'Kitti3dDataset'
data_root = 'data/kitti/'
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
    #    type='RepeatDataset',
    #    times=10,
    #    dataset=dict(
        type=dataset_type,
        handcraft_feats=False,
        load_feats=True,
        pc_range=(0, -39.68, -3, 69.12, 39.68, 1),  # c x h x w = 10 x 496 x 432
        resolution = (0.16, 0.16, 0.4),
        ann_file=data_root + 'ImageSets/train.txt',
        pc_prefix=data_root + 'training/pointpillars_496x432_Px100x9_np16/'),
    #),
    val=dict(
        type=dataset_type,
        handcraft_feats=False,
        load_feats=True,
        pc_range=(0, -39.68, -3, 69.12, 39.68, 1),  # h x w = 496 x 432
        resolution = (0.16, 0.16, 0.4),
        ann_file=data_root + 'ImageSets/val.txt',
        pc_prefix=data_root + 'training/pointpillars_496x432_Px100x9_np16/'),
    test=dict(
        type=dataset_type,
        handcraft_feats=False,
        load_feats=True,
        pc_range=(0, -39.68, -3, 69.12, 39.68, 1),  # h x w = 496 x 432
        resolution = (0.16, 0.16, 0.4),
        ann_file=data_root + 'ImageSets/val.txt',
        pc_prefix=data_root + 'training/pointpillars_496x432_Px100x9_np16/'))
# optimizer
optimizer = dict(type='Adam', lr=0.0002, weight_decay=1e-4)   # 0.0004
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    gamma = 0.8,
    step=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 160
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pillar_kitti_focalloss'
load_from = None
resume_from = None
workflow = [('train', 1)]

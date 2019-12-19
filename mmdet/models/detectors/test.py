from rpn_det import RPNDet
from pc_utils import *

net = RPNDet()

print(net)

anchor_gen = AnchorGeneratorStride()

print(anchor_gen.class_name)
print(anchor_gen.num_anchors_per_localization)
print(anchor_gen.ndim)
print(anchor_gen.custom_ndim)

ans = anchor_gen.generate((3, 4, 5))
print(ans)
print(ans.shape)

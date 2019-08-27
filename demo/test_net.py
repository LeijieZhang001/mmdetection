
from mmcv.cnn import VGG
import torch

#net = VGG(11, with_bn=True, num_stages=3, dilations=(1,1,1), out_indices=(0,1,2))
net = VGG(11, with_bn=True)
net.forward(torch.zeros([1,3,224,224]))

print(net)

print(net.state_dict().keys())

checkpoint = torch.load("/home/leijie/coding/mmdetection/work_dirs/ssd300_voc/latest.pth", map_location=torch.device("cpu"))
print(checkpoint.keys())
#print(checkpoint['meta'])
print(checkpoint['state_dict'].keys())
print('----')
print(checkpoint['optimizer'].keys())
print(checkpoint['optimizer']['param_groups'])

print('++++')
print(list(checkpoint['state_dict'].keys())[0])

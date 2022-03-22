from dis import dis
from numpy import logical_and
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

import sys, os

sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet
from util.classification_utils import extract_mask_distributions

class PSPNetContext(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, label_res_factor=1, in_size=512, pspnet_weights=None):
        super(PSPNetContext, self).__init__()
        self.pspnet = PSPNet(layers=layers, classes=classes, zoom_factor=8, pretrained=False, ppm_frac=8)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(fix_state_dict, strict=True)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.label_res_factor = label_res_factor
        self.classes = classes
        self.upsample = nn.Upsample(size=(512 // label_res_factor, 512 // label_res_factor), mode='bilinear', align_corners=False)
        # self.label_downsample = nn.AdaptiveAvgPool2d(output_size=(512 // label_res_factor, 512 // label_res_factor))

    def forward(self, x, y=None):
        """
        y[0] = segmentation label
        y[1] = distribution label
        distributions = GT distributions for inference
        """
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.layer0(x)
        x = self.pspnet.layer1(x)
        x = self.pspnet.layer2(x)
        x = self.pspnet.layer3(x)
        x_alt = self.pspnet.aux(x)
        x = self.pspnet.layer4(x)

        x = self.pspnet.ppm(x)

        # fix fraction here just for computation calculation

        x = self.pspnet.cls(x)

        x = self.upsample(x)
        x_alt = self.upsample(x_alt)

        return x, x_alt

if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    model = PSPNetContext()
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # feature map 1/8 - 194.81 GMac
    # feature map 1/4 - 427.75 GMac

    from size_estimine import SizeEstimator
    
    se = SizeEstimator(model, input_size=(16, 3, 512, 512))
    print(se.calc_forward_backward_bits())

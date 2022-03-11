from distutils.command.config import config
from multiprocessing import pool
from tkinter.tix import Tree
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.ops import resize

resnet_config = "exp/ade20k/upernet50/model/config.py"
resnet_checkpoint = "exp/ade20k/upernet50/model/upernet_r50.pth"
swin_config = "exp/ade20k/upernet_swin/model/config.py"
swin_checkpoint = "exp/ade20k/upernet_swin/model/upernet_swin_t.pth"

class FCNHead(nn.Module):
    def __init__(self, in_features=512, pool=False, sigmoid=False):
        super(FCNHead, self).__init__()
        self.layer1 = nn.Conv2d(in_features, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout2d(p=0.1)
        self.layer2 = nn.Conv2d(256, 150, kernel_size=1, bias=True)
        self.sigmoid = sigmoid
        self.pool = pool

    def forward(self, x):
        if self.pool:
            x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = self.layer1(x)
        x = self.bn(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.layer2(x)
        if self.sigmoid:
            x = nn.Sigmoid()(x)
        return x

class UPerNet(nn.Module):
    def __init__(self, backbone="resnet", aux_class=False, init_weights=None, pretrained=False):
        super(UPerNet, self).__init__()
        assert backbone in ["resnet", "swin"]
        config = resnet_config if backbone == "resnet" else swin_config
        checkpoint = resnet_checkpoint if backbone == "resnet" else swin_checkpoint
        if not pretrained:
            checkpoint = None
        model = init_segmentor(config, checkpoint, device='cuda:0')
        self.backbone = model.backbone
        self.decode_head = model.decode_head
        self.aux_class = aux_class
        if not self.aux_class:
            self.aux_head = FCNHead(in_features=1024, pool=False, sigmoid=False)  # for stage4 resnet segmentation
        else:
            self.aux_head = FCNHead(in_features=1024, pool=True, sigmoid=True)  # for stage4 resnet classification        
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.aux_loss = nn.BCELoss() if aux_class else nn.CrossEntropyLoss(ignore_index=255)

        if init_weights is not None:
            checkpoint = torch.load(init_weights)["state_dict"]
            self.load_state_dict(checkpoint, strict=False)

    def upernet_forward(self, inputs):
        inputs = self.decode_head._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.decode_head.lateral_convs)
        ]
        laterals.append(self.decode_head.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.decode_head.align_corners)

        # build outputs
        fpn_outs = [
            self.decode_head.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        pre_cls = self.decode_head.fpn_bottleneck(fpn_outs)
        logits = self.decode_head.cls_seg(pre_cls)
        return logits

    def forward(self, x, y=None):
        h, w = x.size()[2:] 
        stages = self.backbone(x)
        x = self.upernet_forward(stages)
        x = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=self.decode_head.align_corners)

        # faster inference
        if y is None:
            return x

        # collect losses
        seg_loss = self.seg_loss(x, y[0])
        # auxiliary prediction
        aux_pred = self.aux_head(stages[2])
        if self.aux_class:  # classification
            aux_loss = self.seg_loss(aux_pred, y[1])
        else:
            aux_pred = F.interpolate(aux_pred, size=(h,w), mode="bilinear", align_corners=self.decode_head.align_corners)
            aux_loss = self.seg_loss(aux_pred, y[0])

        return x, seg_loss, aux_loss


if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 512, 512)).cuda()
    model = UPerNet(backbone="resnet")
    pred = model.forward(x)
    print("hello")
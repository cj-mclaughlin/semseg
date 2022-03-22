from cProfile import label
from distutils.command.config import config
from multiprocessing import pool
from socket import ALG_OP_DECRYPT
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
    def __init__(self, backbone="resnet", aux_class=False, init_weights=None, pretrained=False, fm_scale=4, pool=False):
        super(UPerNet, self).__init__()
        assert backbone in ["resnet", "swin"]
        config = resnet_config if backbone == "resnet" else swin_config
        checkpoint = resnet_checkpoint if backbone == "resnet" else swin_checkpoint
        if not pretrained:
            checkpoint = None
        model = init_segmentor(config, checkpoint, device='cpu')
        self.backbone = model.backbone
        self.decode_head = model.decode_head
        self.aux_class = aux_class
        self.aux_head = FCNHead(in_features=1024, pool=False, sigmoid=False)
        # if not self.aux_class:
        #     self.aux_head = FCNHead(in_features=1024, pool=False, sigmoid=False)  # for stage4 resnet segmentation
        # else:
        #     self.aux_head = FCNHead(in_features=1024, pool=False, sigmoid=True)  # for stage4 resnet classification        
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.aux_loss = nn.BCELoss() # if aux_class else nn.CrossEntropyLoss(ignore_index=255)

        self.fusion_upsample = nn.Upsample(size=(512 // fm_scale), mode="bilinear", align_corners=False)

        if init_weights is not None:
            checkpoint = torch.load(init_weights)["state_dict"]
            self.load_state_dict(checkpoint, strict=False)

    def upernet_forward(self, inputs, pool):
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

        for i in range(len(fpn_outs)):
            fpn_outs[i] = self.fusion_upsample(fpn_outs[i])
        fpn_outs = torch.cat(fpn_outs, dim=1)
        pre_cls = self.decode_head.fpn_bottleneck(fpn_outs)
        if pool:
            pre_cls = nn.AdaptiveAvgPool2d(output_size=(1,1))(pre_cls)
        logits = self.decode_head.cls_seg(pre_cls)
        return logits

    def forward(self, x, y=None):
        h, w = x.size()[2:]
        # seg_label = y[0]
        # cls_label = y[1]
        self.eval()
        stages = self.backbone(x)
        seg_x = self.upernet_forward(stages, pool=False)

        # losses = []
        # for label_scale in [1, 2, 4, 8]:
        #     x = F.interpolate(seg_x, size=(h // label_scale, w // label_scale), mode="bilinear", align_corners=False)
        #     y = F.interpolate(seg_label.unsqueeze(1).float(), size=(h // label_scale, w // label_scale), mode="nearest")[:,0].long()
        #     losses.append(self.seg_loss(x, y))
        
        # cls_x = self.upernet_forward(stages, pool=True)
        # cls_x = nn.Sigmoid()(cls_x)
        # losses.append(self.aux_loss(cls_x, cls_label))

        return seg_x        
        # # faster inference
        # if y is None:
        #     return x

        # # collect losses
        # seg_loss = self.seg_loss(x, y[0])
        # # auxiliary prediction
        # aux_pred = self.aux_head(stages[2])
        # if self.aux_class:  # classification
        #     aux_loss = self.seg_loss(aux_pred, y[1])
        # else:
        #     aux_pred = F.interpolate(aux_pred, size=(h,w), mode="bilinear", align_corners=self.decode_head.align_corners)
        #     aux_loss = self.seg_loss(aux_pred, y[0])

        # return x, seg_loss, aux_loss


if __name__ == "__main__":
    # for each feature map size, compute mem MACs
    for fm_scale in [4, 8, 16, 32]:
        from ptflops import get_model_complexity_info
        model = UPerNet(backbone="resnet", fm_scale=fm_scale)
        macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
        print(f"Statistics for feature map scale {fm_scale}")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # for each feature map size, compute gradient at several locations w.r.t. [1, 2, 4, 8, 16] scale segmentation, as well as classification
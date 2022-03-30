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

class ConvBnRelu(nn.Module):
    def __init__(self, in_features=256, out_features=256, dropout=0.1):
        super(ConvBnRelu, self).__init__()
        self.layer1 = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = nn.ReLU()(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class JointPrediction(nn.Module):
    def __init__(self, in_features=256, classes=150):
        super(JointPrediction, self).__init__()
        self.layer1 = nn.Conv2d(in_features, classes, kernel_size=1, bias=True)

    def forward(self, x):
        classification = nn.AdaptiveAvgPool2d((1,1))(x)
        classification = self.layer1(classification)
        x = self.layer1(x)
        return x, classification

class UPerNet(nn.Module):
    def __init__(self, backbone="resnet", init_weights=None, pretrained=False):
        super(UPerNet, self).__init__()
        assert backbone in ["resnet", "swin"]
        config = resnet_config if backbone == "resnet" else swin_config
        checkpoint = resnet_checkpoint if backbone == "resnet" else swin_checkpoint
        if not pretrained:
            checkpoint = None
        model = init_segmentor(config, checkpoint, device='cpu')
        self.backbone = model.backbone
        self.decode_head = model.decode_head

        self.segmentation_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.classification_loss = nn.BCEWithLogitsLoss()

        self.dropout = nn.Dropout2d(p=0.1)

        self.aux_bottleneck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        
        self.aux_prediction = JointPrediction(in_features=256, classes=150)
        self.main_prediction = JointPrediction(in_features=512, classes=150)

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
        outputs = [
            self.decode_head.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        outputs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            outputs[i] = resize(
                outputs[i],
                size=outputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.decode_head.fpn_bottleneck(outputs)
        outputs = self.dropout(outputs)
        return outputs

    def forward(self, x, y=None):
        h, w = x.size()[2:]

        stages = self.backbone(x)
        x = self.upernet_forward(stages)

        main_seg, main_class = self.main_prediction(x)
        main_seg = F.interpolate(main_seg, size=(h, w), mode="bilinear", align_corners=False)

        if self.training: 
            aux = self.aux_bottleneck(stages[-2])
            aux_seg, aux_class = self.aux_prediction(aux)
            aux_seg = F.interpolate(aux_seg, size=(h,w), mode="bilinear", align_corners=False)

            class_label = _convert_to_onehot_labels(y, num_classes=150)
            aux_loss = [self.segmentation_loss(aux_seg, y), self.classification_loss(torch.squeeze(aux_class), class_label)]
            main_loss = [self.segmentation_loss(main_seg, y), self.classification_loss(torch.squeeze(main_class), class_label)]

            return main_seg, main_loss, aux_loss

        else:
            return main_seg   


def _convert_to_onehot_labels(seg_label, num_classes):
    """Convert segmentation label to onehot.
    Args:
        seg_label (Tensor): Segmentation label of shape (N, H, W).
        num_classes (int): Number of classes.
    Returns:
        Tensor: Onehot labels of shape (N, num_classes).
    """

    batch_size = seg_label.size(0)
    onehot_labels = seg_label.new_zeros((batch_size, num_classes))
    for i in range(batch_size):
        hist = seg_label[i].float().histc(
            bins=num_classes, min=0, max=num_classes - 1)
        onehot_labels[i] = hist > 0
    return onehot_labels.float()


if __name__ == "__main__":
    model = UPerNet()
    x = torch.rand(size=(8, 3, 512, 512))
    y = [torch.rand(size=(8, 512, 512)), torch.rand(size=(8, 150, 512, 512))]
    model.forward(x, y)

    # for each feature map size, compute mem MACs
    # for fm_scale in [4, 8, 16, 32]:
    #     from ptflops import get_model_complexity_info
    #     model = UPerNet(backbone="resnet", fm_scale=fm_scale)
    #     macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    #     print(f"Statistics for feature map scale {fm_scale}")
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # for each feature map size, compute gradient at several locations w.r.t. [1, 2, 4, 8, 16] scale segmentation, as well as classification
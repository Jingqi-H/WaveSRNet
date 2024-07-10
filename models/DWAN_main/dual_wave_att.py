import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math

from thop import profile
from thop import clever_format
import torch.nn as nn
# from dwtmodel.DWT_IDWT.DWT_IDWT_layer import *
from models.DWAN_main.dwtmodel.DWT_IDWT.DWT_IDWT_layer import *
import torch


"""
框架粘贴自coord，dual wave att 来自官方代码

"""

class Downsamplewave(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        LL,LH,HL,HH = self.dwt(input)
        return torch.cat([LL,LH+HL+HH],dim=1)

class Downsamplewave1(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave1, self).__init__()
        # self.dwt = DWT_2D_tiny(wavename = wavename) #return LL
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        # LL = self.dwt(input)
        # return LL
        # inputori= input
        LL,LH,HL,HH = self.dwt(input)
        LL = LL+LH+HL+HH
        result = torch.sum(LL, dim=[2, 3])  # x:torch.Size([64, 256, 56, 56])
        return result  ###torch.Size([64, 256])

class Waveletatt(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave1(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        # x0,x1,x2,x3 = Downsamplewave(x)
        ##x0,x1,x2,x3= self.downsamplewavelet(x)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y

class Waveletattspace(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            # nn.Linear(in_planes, in_planes // 2, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes,kernel_size=1,padding= 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        y = self.fc(y) # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        # y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)
        return y

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, deep_dim=18):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 2023.8.2
        if deep_dim == 18:
            self.att1 = Waveletatt(in_planes=64)
            self.attspace1 = Waveletattspace(in_planes=64)
            self.att2 = Waveletatt(in_planes=128)
            self.attspace2 = Waveletattspace(in_planes=128)
            self.att3 = Waveletatt(in_planes=256)
            self.att4 = Waveletatt(in_planes=512)
        else:
            self.att1 = Waveletatt(in_planes=256)
            self.attspace1 = Waveletattspace(in_planes=256)
            self.att2 = Waveletatt(in_planes=512)
            self.attspace2 = Waveletattspace(in_planes=512)
            self.att3 = Waveletatt(in_planes=1024)
            self.att4 = Waveletatt(in_planes=2048)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        # 2023.8.2
        x = self.att2(x)
        x = self.attspace2(x)

        x = self.layer3(x)
        # x = self.att3(x)

        x = self.layer4(x)
        # x = self.att4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dwa_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model




def dwa_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_dim=50, **kwargs)
    return model

# def dwa_resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model
#
# def dwa_resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def dwa_resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model


if __name__ == '__main__':
    import os
    from thop import profile
    from thop import clever_format

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    temp = torch.randn((26, 1, 224, 224))   # res50+dwa 最多只能26，在main中跑不通，最后用了16
    # net = dwa_resnet18(num_classes=2)
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # net.cuda()
    # y = net(temp.cuda())
    # print(y.shape)

    net = dwa_resnet50(num_classes=2)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    net.cuda()
    y = net(temp.cuda())
    print(y.shape)

    # from torchstat import stat
    # stat(net.cpu(), (3, 224, 224))


    # macs, params = profile(net, inputs=(torch.randn([2,1,224,224]).cuda(),))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('macs:', macs, 'params:', params)
#     都没有：macs: 31.989G params: 23.503M
#     只有att2：macs: 31.990G params: 23.765M
#     有att2和attspace2：macs: 34.461G params: 24.159M


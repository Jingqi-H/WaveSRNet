import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math


# __all__ = ['se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']


class MSAPooling(nn.Module):
    # also called RCR
    def __init__(self, nchannels):
        super(MSAPooling, self).__init__()
        self.nchannels = nchannels

        # paramters
        self.cfc = Parameter(torch.Tensor(self.nchannels, 3))
        self.cfc.data.fill_(0)

        self.cfc_avg = Parameter(torch.Tensor(self.nchannels, 3))
        self.cfc_avg.data.fill_(1)

        self.cfc_max = Parameter(torch.Tensor(self.nchannels, 3))
        self.cfc_max.data.fill_(0)

        self.cfc_std = Parameter(torch.Tensor(self.nchannels, 3))
        self.cfc_std.data.fill_(0)

        setattr(self.cfc, 'srm_param', True)
        setattr(self.cfc_avg, 'srm_param', True)
        setattr(self.cfc_max, 'srm_param', True)
        setattr(self.cfc_std, 'srm_param', True)
        # pooling operations  

        self.sigmoid = nn.Sigmoid()

        self.bn = nn.BatchNorm2d(self.nchannels)
        #  build  connections among  feature representations and channels

        # self.channel_conv = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def _max_pooling(self, x):
        N, C, height, width = x.size()
        x1 = x[:, :, :height // 2, :]  # up region
        x2 = x[:, :, height // 2:, :]  # bottom region

        max, idx = x.view(N, C, -1).max(dim=2, keepdim=True)

        max_up, idx = x1.view(N, C, -1).max(dim=2, keepdim=True)
        max_bottom, idx = x1.view(N, C, -1).max(dim=2, keepdim=True)

        t = torch.cat((max, max_up, max_bottom), dim=2)
        return t

    def _max_integration(self, t):
        z = t * self.cfc_max[None, :, :]  # B x C x 3
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        z_max = self.bn(z)
        return z_max

    def _avg_pooling(self, x):
        N, C, height, width = x.size()
        x1 = x[:, :, :height // 2, :]  # up region
        x2 = x[:, :, height // 2:, :]  # bottom region

        mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        mean_bottom = x2.view(N, C, -1).mean(dim=2, keepdim=True)
        mean_up = x1.view(N, C, -1).mean(dim=2, keepdim=True)

        t = torch.cat((mean, mean_up, mean_bottom), dim=2)
        return t

    def _avg_integration(self, t):
        z = t * self.cfc_avg[None, :, :]  # B x C x 3
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_avg = self.bn(z)

        return z_avg

    def _std_pooling(self, x, eps=1e-5):
        N, C, height, width = x.size()
        x1 = x[:, :, :height // 2, :]  # up region
        x2 = x[:, :, height // 2:, :]  # bottom region

        var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        std = var.sqrt()

        var_up = x1.view(N, C, -1).var(dim=2, keepdim=True) + eps
        std_up = var_up.sqrt()

        var_bottom = x2.view(N, C, -1).var(dim=2, keepdim=True) + eps
        std_bottom = var_bottom.sqrt()

        t = torch.cat((std, std_up, std_bottom), dim=2)
        return t

    def _std_integration(self, t):
        z = t * self.cfc_std[None, :, :]  # B x C x 3
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_std = self.bn(z)

        return z_std

    def _fuse_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 3
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_fuse = self.bn(z)

        return z_fuse

    def forward(self, x):
        # input tensor shape
        # B x C x 3
        N, C, height, width = x.size()

        max_t = self._max_pooling(x)
        max_v = self._max_integration(max_t)

        avg_t = self._avg_pooling(x)
        avg_v = self._avg_integration(avg_t)

        std_t = self._std_pooling(x)
        std_v = self._std_integration(avg_t)

        t = torch.cat((max_v.view(N, C, -1), avg_v.view(N, C, -1),
                       std_v.view(N, C, -1)), dim=2)
        fuse_v = self._fuse_integration(t)

        g_fuse = self.sigmoid(fuse_v)

        return x * g_fuse


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
        self.se = MSAPooling(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

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
        self.se = MSAPooling(planes * self.expansion)
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
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3, zero_init_residual=False):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def rcr_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def rcr_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def rcr_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def rcr_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def rcr_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    temp = torch.randn(64, 1, 224, 224).cuda()
    net = rcr_resnet18(num_classes=2)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.cuda()
    y = net(temp)
    print(y.size())

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(torch.randn((2, 1, 224, 224)).cuda(),))
    macs, params = clever_format([macs, params], "%.3f")
    print('macs:', macs, 'params:', params)

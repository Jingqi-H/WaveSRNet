import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavecnet_DWT_IDWT.DWT_IDWT_layer import DWT_2D_tiny

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
model_urls = {
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
 }

class Downsample(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL



def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, wavename = 'haar'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride==1):
            self.conv2 = conv3x3(planes,planes)
        else:
            self.conv2 = nn.Sequential(Downsample(wavename = wavename),
                conv3x3(planes, planes),)
        self.bn2 = norm_layer(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, wavename = 'haar'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups) # stride moved
        self.bn2 = norm_layer(planes)
        if(stride==1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample(wavename = wavename),
                conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, wavename = 'haar', pool_only = True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            self.maxpool = nn.Sequential(*[Downsample(wavename = wavename)])
        else:
            self.maxpool = nn.Sequential(*[Downsample(wavename = wavename), Downsample(wavename = wavename)])

        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, wavename = 'haar'):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [Downsample(wavename = wavename),] if(stride == 2) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, wavename = wavename))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, wavename = wavename))

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


def resnet18(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet34(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet50(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet101(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet152(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnext50_32x4d(wavename = 'haar', pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, wavename = wavename, pool_only=pool_only, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(wavename = 'haar', pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, wavename = wavename, pool_only=pool_only, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

if __name__ == '__main__':
    #  CUDA_VISIBLE_DEVICES=0 python main.py --data /PYTHON_TO_IMANGENET -a resnet18_dwt -b 256 -w bior3.3 --gpu 0 --lr 0.1
    import torch
    net = resnet18(num_classes=2)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.cuda()
    print(net)

    temp = torch.randn((64, 1, 224, 224)).cuda()
    y = net(temp)
    print(y.shape)

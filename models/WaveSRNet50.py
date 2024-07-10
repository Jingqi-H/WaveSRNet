import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math

# from custom_wavelet import DWT
from models.custom_wavelet import DWT
from models.scSE_recalibration import ChannelSpatialSELayer, ChannelSELayer, SpatialSELayer


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0 \
                 , dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size \
                              , stride=stride, padding=padding, dilation=dilation \
                              , groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01 \
                                 , affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Custom_wave(nn.Module):
    def __init__(self, in_planes):
        super(Custom_wave, self).__init__()
        self.pad = nn.ReflectionPad2d((1,0,1,0))  # 左边和上边padding 1行
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.enc = nn.Sequential(
            nn.Linear(in_planes, 1),
            nn.Sigmoid()
        )

        self.mm = DWT()  # from DualWaveletAttention
        self.mm2 = DWT()  # from DualWaveletAttention

    def dwt_custom1(self, x):
        n, c, h, w = x.shape
        if h % 2 == 1:
            x = self.pad(x)
        x_LL, x_HL, x_LH, x_HH = self.mm(x)
        return x_LL, x_HL, x_LH, x_HH

    def selection(self, inputs, option='max'):
        # https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
        # inputs [bs, num_FrequencyBand, num_feats]
        A1 = []
        Q = []
        bs = inputs.shape[0]
        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):  # 循环bs，每个i是4个频段的特征向量[num_FrequencyBand,num_feats]
                # 1 https://github.com/PhilipChicco/FRMIL/blob/main/models/mil_ss.py
                # print('qwe:', inputs[i].shape) # torch.Size([4, 64])
                # print(inputs[i].unsqueeze(0).shape)  # torch.Size([1, 4, 64])
                # print(self.enc)
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                # print('qwe:', a1.shape)  # torch.Size([4, 1])
                # 2 CLAM的attention
                # a1,aa = self.att(inputs[i].unsqueeze(0))
                # a1 = a1.squeeze(0)
                # print('a1:', a1.shape)  # torch.Size([10, 1])

                _, m_indices = torch.sort(a1, 0, descending=True)  # 每个频段的重要性排序
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(1): # self.k=1
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                        # print('torch.index_select',feats.shape)
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)

                feats = torch.stack(feat_q)

                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))

            A1 = torch.stack(A1)
            Q = torch.stack(Q)
            return A1, Q

    def forward(self, input):
        LL, LH, HL, HH = self.dwt_custom1(input)  # ([64, 64, 28, 28])

        # selection1
        # ori_y = self.avgpool(torch.stack([LL, LH, HL, HH], dim=1))  #和torch.mean(dim=[3,4])效果一致
        # y = torch.sum(ori_y, dim=1)

        # selection2
        # print('after dwt_custom1:',LL.shape,LH.shape,HL.shape,HH.shape)
        ori_y = self.avgpool(torch.stack([LL, LH, HL, HH], dim=1))  #LL, LH, HL, HH。和torch.mean(dim=[3,4])效果一致
        # ori_y = torch.sum(torch.stack([LL, LH, HL, HH], dim=1),dim=[3,4])  #LL, LH, HL, HH。和torch.mean(dim=[3,4])效果一致
        # print('after self.avgpool:',ori_y.shape)
        y_before = ori_y.squeeze()
        # print('after self.avgpool:',ori_y.shape)
        # print('y_before:',y_before.shape)
        A1, Q = self.selection(y_before, 'max')
        # print('after selection')
        # print('A1:',A1.shape)  # torch.Size([64, 1, 64])
        # print('qqq:',Q.shape)  # torch.Size([64, 1, 64])
        y_after = F.relu(y_before - Q)
        # print('y_after:',y_after.shape)  # torch.Size([64, 1, 64])
        # 1
        y = torch.sum(y_after, dim=1)
        # 2
        # y = torch.mean(y_after, dim=1)
        # print('after torch.sum:',y.shape)  # torch.Size([64, 1, 64])

        # selection3
        # y1 = self.avgpool(torch.stack([LL, LH, HL, HH], dim=1))
        # y2 = self.avgpool(torch.stack([LL2, LH2, HL2, HH2], dim=1))
        # ori_y = torch.concat([y1, y2], dim=1)
        # y_before = ori_y.squeeze()
        # A1, Q = self.selection(y_before, 'max')
        # # print('qqq',Q.shape)  # torch.Size([64, 1, 64])
        # y_after = F.relu(y_before - Q)
        # # 1
        # # y = torch.sum(y_after, dim=1)
        # # 2
        # y = torch.mean(y_after, dim=1)

        return y  ###torch.Size([64, 256])


class Waveletatt(nn.Module):
    def __init__(self, in_planes, dct_h, dct_w, input_resolution=224, ):
        super(Waveletatt, self).__init__()
        self.input_resolution = input_resolution
        self.dct_h = dct_h
        self.dct_w = dct_w

        self.dct_layer = Custom_wave(in_planes)

        self.fc = nn.Sequential(

            # 通道注意力
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()

            # nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

        # self.recalibration = ChannelSpatialSELayer(in_planes)# csSE
        # self.recalibration = SpatialSELayer(in_planes) # sSE
        self.recalibration = ChannelSELayer(in_planes) # cSE
        # self.recalibration = SRMLayer(in_planes) # SMR


    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        # print('x_pooled:',x_pooled.shape)
        y = self.dct_layer(x_pooled)  # torch.Size([8, 64])
        # print('after selection:',y.shape)


        y = self.fc(y.squeeze()).view(n, c, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        # y = self.fc(y.view(n, c, 1, 1)  )# torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        # print('after two fc:',y.shape)

        # recalibration
        y = self.recalibration(y)
        # x_hat = self.recalibration(x_hat)
        # print('after recalibration:',y.shape)
        # print('P bolang:',(x * y.expand_as(x)).shape)
        # asdf

        return x * y.expand_as(x)  # re-weight





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_wave_in=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        if use_wave_in:
            self.waveatt = Waveletatt(planes,c2wh[planes], c2wh[planes])
        else:
            self.waveatt = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        # self.waveatt(out)的输出 是reweight之后的特征图
        if not self.waveatt is None:
            out = self.waveatt(out)  # torch.Size([8, 64, 56, 56])-->torch.Size([8, 64, 56, 56])

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_wave_in=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])

        if use_wave_in:
            self.waveatt = Waveletatt(planes*4, c2wh[planes], c2wh[planes])
        else:
            self.waveatt = None


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.waveatt is None:
            out = self.waveatt(out)

        out += residual
        out = self.relu(out)

        return out



class Waveletatt_afterlayer(nn.Module):
    def __init__(self, in_planes):
        super(Waveletatt_afterlayer, self).__init__()

        self.fc = nn.Sequential(

            # 通道注意力
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()

            # nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

        # self.recalibration = ChannelSpatialSELayer(in_planes)# csSE
        # self.recalibration = SpatialSELayer(in_planes) # sSE
        self.recalibration = ChannelSELayer(in_planes) # cSE
        # self.recalibration = SRMLayer(in_planes) # SMR

        self.pad = nn.ReflectionPad2d((1, 0, 1, 0))  # 左边和上边padding 1行
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.enc = nn.Sequential(
            nn.Linear(in_planes, 1),
            nn.Sigmoid()
        )

        self.mm = DWT()  # from DualWaveletAttention


    def dwt_custom1(self, x):
        n, c, h, w = x.shape
        if h % 2 == 1:
            x = self.pad(x)
        x_LL, x_HL, x_LH, x_HH = self.mm(x)
        return x_LL, x_HL, x_LH, x_HH


    def selection(self, inputs, option='max'):
        # https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
        # inputs [bs, num_FrequencyBand, num_feats]
        A1 = []
        Q = []
        bs = inputs.shape[0]
        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):  # 循环bs，每个i是4个频段的特征向量[num_FrequencyBand,num_feats]
                # 1 https://github.com/PhilipChicco/FRMIL/blob/main/models/mil_ss.py
                # print('qwe:', inputs[i].shape) # torch.Size([4, 64])
                # print(inputs[i].unsqueeze(0).shape)  # torch.Size([1, 4, 64])
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                # print('qwe:', a1.shape)  # torch.Size([4, 1])
                # 2 CLAM的attention
                # a1,aa = self.att(inputs[i].unsqueeze(0))
                # a1 = a1.squeeze(0)
                # print('a1:', a1.shape)  # torch.Size([10, 1])

                _, m_indices = torch.sort(a1, 0, descending=True)  # 每个频段的重要性排序
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(1):  # self.k=1
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                        # print('12',feats.shape)
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)

                feats = torch.stack(feat_q)

                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))

            A1 = torch.stack(A1)
            Q = torch.stack(Q)
            return A1, Q

    def forward(self, x):
        n, c, h, w = x.shape
        ori_x = x
        LL, LH, HL, HH = self.dwt_custom1(x)  # ([64, 64, 28, 28])

        # selection
        ori_y = self.avgpool(torch.stack([LL, LH, HL, HH], dim=1))  # LL, LH, HL, HH。和torch.mean(dim=[3,4])效果一致
        y_before = ori_y.squeeze()
        A1, Q = self.selection(y_before, 'max')
        y_after = F.relu(y_before - Q)
        y = torch.sum(y_after, dim=1)
        y = self.fc(y.squeeze()).view(n, c, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])

        # recalibration
        y = self.recalibration(y)

        return ori_x * y.expand_as(ori_x)  # re-weight


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == 'Wave':
            self.wave0 = Waveletatt_afterlayer(64)
            self.wave1 = Waveletatt_afterlayer(64)
            self.wave2 = Waveletatt_afterlayer(128)
            self.wave3 = Waveletatt_afterlayer(256)
            self.wave4 = Waveletatt_afterlayer(512)
            # self.wave2, self.wave3 = None, None
        else:
            self.wave1, self.wave2, self.wave3 = None, None, None

        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_wave_in=att_type == 'Wave_in'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_wave_in=att_type == 'Wave_in'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)
        # if not self.wave0 is None:
        #     x = self.wave0(x)
        # print(x.shape)

        x = self.layer1(x)
        if not self.wave1 is None:
            x = self.wave1(x)
        # print(x.shape)

        x = self.layer2(x)
        # if not self.wave2 is None:
        #     x = self.wave2(x)
        # print(x.shape)

        x = self.layer3(x)
        # if not self.wave3 is None:
        #     x = self.wave3(x)
        # print(x.shape)

        x = self.layer4(x)
        # if not self.wave4 is None:
        #     x = self.wave4(x)
        # print(x.shape)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type=None):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


# def wave_resnet18(pretrained=False, **kwargs):
#     model = ResidualNet("ImageNet", 18, 2, 'CBAM')
#     return model

def wave_resnet18(pretrained=False, num_classes=2,**kwargs):
    model = ResidualNet(network_type="ImageNet", depth=18, num_classes=num_classes, att_type='Wave_in') # Wave_in Wave
    return model


# def wave_resnet34(pretrained=False, **kwargs):
#     model = ResidualNet("ImageNet", 34, 2, 'CBAM')
#     return model


def wave_resnet50(pretrained=False, num_classes=2, **kwargs):
    model = ResidualNet(network_type="ImageNet", depth=50, num_classes=num_classes, att_type='Wave_in')
    return model


# def cbam_resnet101(pretrained=False, **kwargs):
#     model = ResidualNet('ImageNet', 101, 1000, 'CBAM')
#     return model


if __name__ == '__main__':
    temp = torch.randn((2, 1, 224, 224)).cuda()
    # net = wave_resnet18(num_classes=3, levels=4)
    net = wave_resnet18(num_classes=3, levels=4)
    # torch.Size([2, 64, 56, 56])
    # torch.Size([2, 256, 56, 56])
    # torch.Size([2, 512, 28, 28])
    # torch.Size([2, 1024, 14, 14])
    # torch.Size([2, 2048, 7, 7])

    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.cuda()

    y = net(temp)
    # print(net)
    print(y.shape)

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net, inputs=(temp, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('macs:', macs,'params:', params)
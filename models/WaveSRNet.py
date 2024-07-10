import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

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
    def __init__(self, in_planes,wavename = 'haar'):
        super(Custom_wave, self).__init__()
        self.pad = nn.ReflectionPad2d((1,0,1,0))  # 左边和上边padding 1行
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.enc = nn.Sequential(
            nn.Linear(in_planes, 1),
            nn.Sigmoid()
        )

        self.mm = DWT(wavename)

    def dwt_custom(self, x):
        n, c, h, w = x.shape
        if h % 2 == 1:
            x = self.pad(x)
        x_LL, x_HL, x_LH, x_HH = self.mm(x)
        return x_LL, x_HL, x_LH, x_HH


    def recalib(self, inputs, option='max'):
        A1 = []
        Q = []
        bs = inputs.shape[0]

        if bs == 4:
            inputs = inputs.unsqueeze(dim=0)
            bs = inputs.shape[0]

        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                _, m_indices = torch.sort(a1, 0, descending=True)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(1):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
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
        LL, LH, HL, HH = self.dwt_custom(input)

        ori_y = self.avgpool(torch.stack([LL, LH, HL, HH], dim=1))
        y_before = ori_y.squeeze()
        A1, Q = self.recalib(y_before, 'max')
        y_after = F.relu(y_before - Q)
        y = torch.sum(y_after, dim=1)

        return y, A1, ori_y, y_after, torch.stack([LL, LH, HL, HH], dim=0)


class Waveletatt(nn.Module):
    def __init__(self, in_planes, dct_h, dct_w, input_resolution=224, ):
        super(Waveletatt, self).__init__()
        self.input_resolution = input_resolution
        self.dct_h = dct_h
        self.dct_w = dct_w

        self.dct_layer = Custom_wave(in_planes)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )


        self.csSE = ChannelSpatialSELayer(in_planes)
        self.sSE = SpatialSELayer(in_planes)
        self.cSE = ChannelSELayer(in_planes)


    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y,_,_,_,_ = self.dct_layer(x_pooled)  # torch.Size([8, 64])


        y1 = self.fc(y.squeeze()).view(n, c, 1, 1)
        y2 = self.cSE(y1)

        return x * y2.expand_as(x), y, y2


def conv3x3(in_planes, out_planes, stride=1):
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

        if not self.waveatt is None:
            out,_,_ = self.waveatt(out)

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


        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
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
            out,_,_  = self.waveatt(out)

        out += residual
        out = self.relu(out)

        return out



class Waveletatt_afterlayer(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        wavename = 'haar'

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.enc = nn.Sequential(
            nn.Linear(in_planes, 1),
            nn.Sigmoid()
        )


    def dwt_custom(self, x):
        n, c, h, w = x.shape
        if h %2 == 1:
            x = self.pad(x)

        x01 = x[:,:, 0::2, :] / 2
        x02 = x[:, :,1::2, :] / 2
        x1 = x01[:, :,:, 0::2]
        x2 = x02[:, :,:, 0::2]
        x3 = x01[:, :,:, 1::2]
        x4 = x02[:, :,:, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, x_HL,x_LH,x_HH

    def recalib(self, inputs, option='max'):
        A1 = []
        Q = []
        bs = inputs.shape[0]
        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                _, m_indices = torch.sort(a1, 0, descending=True)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(1):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
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
        xori = x
        B, C, H, W = x.shape
        LL,LH,HL,HH = self.dwt_custom(x)
        ori_y = self.avgpool(torch.stack([LL,LH,HL,HH],dim=1)).squeeze() #
        A1, Q = self.recalib(ori_y, 'max')
        y = F.relu(ori_y - Q)
        y = torch.sum(y, dim=1)
        y = self.fc(y.squeeze()).view(B, C, 1, 1)
        y = xori * y.expand_as(xori)
        return y



class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == 'Wave':
            self.wave1 = Waveletatt_afterlayer(64)
            self.wave2 = Waveletatt_afterlayer(128)
            self.wave3 = Waveletatt_afterlayer(256)
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
        # if x.size()[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.wave1 is None:
            x = self.wave1(x)

        x = self.layer2(x)
        if not self.wave2 is None:
            x = self.wave2(x)

        x = self.layer3(x)
        # if not self.wave3 is None:
        #     x = self.wave3(x)

        x = self.layer4(x)

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



def wave_resnet18(pretrained=False, num_classes=2,**kwargs):
    model = ResidualNet("ImageNet", 18, num_classes, 'Wave_in') # Wave_in Wave
    return model

def wave_resnet50(pretrained=False, num_classes=2,**kwargs):
    model = ResidualNet("ImageNet", 50, num_classes, 'Wave_in') # Wave_in Wave
    return model


def wave_resnet34(pretrained=False, num_classes=2, **kwargs):
    model = ResidualNet("ImageNet", 34, num_classes, 'Wave_in')
    return model

def wave_resnet101(pretrained=False, num_classes=2, **kwargs):
    model = ResidualNet('ImageNet', 101, num_classes, 'Wave_in')
    return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    temp = torch.randn((64, 1, 224, 224)).cuda()
    net = wave_resnet18(num_classes=3, levels=4)
    # net = wave_resnet34(num_classes=3, levels=4)
    # net = wave_resnet50(num_classes=3, levels=4)
    # net = wave_resnet101(num_classes=3, levels=4)

    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.cuda()

    y = net(temp)
    # # print(net)
    print(y.shape)

    from thop import profile
    from thop import clever_format
    macs, params = profile(net, inputs=(torch.randn((2, 1, 224, 224)).cuda(), ))
    macs, params = clever_format([macs, params], "%.3f")
    print('macs:', macs,'params:', params)
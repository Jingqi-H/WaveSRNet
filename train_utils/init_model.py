import torch.nn as nn
import torchvision.models
import torch
import timm
from collections import OrderedDict
import numpy as np

from models.FcaNet import fcanet
from models.DWAN_main import dual_wave_att
from models.dawn import DAWN
from models.wavepooling import LeNet5
from models.wcnn import WCNN
from models import wave_cnet
from models.oct_methods import resnet_rcr, resnet_rir, FIT_Net
from models import WaveSRNet, WaveSRNet50


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def init_model(model_name, num_classes,pre_train=False):
    if model_name == 'res18':
        # net = torchvision.models.resnet18(num_classes=num_classes)
        net = torchvision.models.resnet18(num_classes=num_classes)
        inchannel = net.fc.in_features  # inchannel=512
        net.fc = nn.Sequential(
            nn.Linear(inchannel, num_classes),
        )
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'res50':
        net = torchvision.models.resnet50(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif model_name == 'dwan':
        net = dual_wave_att.dwa_resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif model_name == 'dwan50':
        net = dual_wave_att.dwa_resnet50(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif model_name == 'fcanet18':
        net = fcanet.fcanet18(pretrained=False, num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name == 'fcanet50':
        net = fcanet.fcanet50(pretrained=False, num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name =='dawn':
        INPUT_SIZE = 32  # 224, 32
        big_input = INPUT_SIZE != 32
        first_conv = 128 # 128
        levels = 3
        kernel_size = 3
        no_bootleneck = False
        classifier = 'mode1'  # ['mode1', 'mode2','mode3']
        share_weights = False
        simple_lifting = False
        monkey = False
        USE_COLOR = not monkey
        regu_details = 0.1
        regu_approx = 0.1
        haar_wavelet = False
        net = DAWN(num_classes, big_input=big_input,
                   first_conv=first_conv,
                   number_levels=levels,
                   kernel_size=kernel_size,
                   no_bootleneck=no_bootleneck,
                   classifier=classifier,
                   share_weights=share_weights,
                   simple_lifting=simple_lifting,
                   COLOR=USE_COLOR,
                   regu_details=regu_details,
                   regu_approx=regu_approx,
                   haar_wavelet=haar_wavelet
                   )
        net.conv1 = nn.Conv2d(1, first_conv, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif model_name == 'wavepooling':
        pooling_type = 'adaptive_wavelet'
        net = LeNet5(pool_type=pooling_type, num_classes=num_classes)
    elif model_name == 'wcnn':
        wavelet = 'haar'
        levels = 4
        INPUT_SIZE = 224  # 224, 32
        big_input = INPUT_SIZE != 32
        net = WCNN(num_classes, big_input=big_input, wavelet=wavelet, levels=levels).cuda()
    elif model_name == 'wavecnet':
        net = wave_cnet.resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name == 'resnet_rir':
        net = resnet_rir.rir_resnet18(num_classes=num_classes)
        net.conv1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'resnet_rcr':
        net = resnet_rcr.rcr_resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'fit_net':
        net = FIT_Net.FITNet_res18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'fit_net_34':
        net = FIT_Net.FITNet_res34(num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'wavesrnet':
        net = WaveSRNet.wave_resnet18(num_classes=num_classes, levels=4)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name == 'ours7back50':
        net = WaveSRNet.wave_resnet50(num_classes=num_classes, levels=4)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_name == 'wavesrnet50':
        net = WaveSRNet50.wave_resnet50(num_classes=num_classes, levels=4)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    else:
        print('model name is none')
        net = None

    return  net

if __name__ == '__main__':
    net = init_model(model_name='wavesrnet', num_classes=2)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    temp = torch.randn((64, 1, 224, 224)).cuda()
    y = net(temp)
    print(y.shape)

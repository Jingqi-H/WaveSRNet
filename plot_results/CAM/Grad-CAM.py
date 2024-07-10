import matplotlib.pyplot as plt
import pylab
import torch
from PIL import Image
from torchvision import transforms
import json
import torch.nn as nn
import os
import torchvision
import time, datetime
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
import random
import sys
import shutil

from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
sys.path.append('/disk1/imed_hjq/code/3-PD_analysis/02_baseline/')
from train_utils.init_model import init_model
from dataloaders.pd_dataset import PD_one_hold_out_Dataset


def img_ten2arr(input_image, imtype=np.uint8):
    """"
    将tensor的数据类型转成numpy类型，并反归一化.
    from https://www.cnblogs.com/wanghui-garcia/p/11393076.html
    Parameters:
        input_image (tensor) --  输入的图像tensor数组 [3, h, w]
        imtype (type)        --  转换后的numpy的数据类型

    :return array格式的img
    """
    mean = [0.485, 0.456, 0.406]  # dataLoader中设置的mean参数
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数
    # mean = [0.5]
    # std = [0.5]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):  # 反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255  # 反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)


def main(root, dataset_factory, model_type, version, model_weight_path, batch_size, num_classes, img_resize,
         img_crop):
    net = init_model(model_type, num_classes, 'cuda')
    net.cuda()

    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    data_transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop((img_crop, img_crop)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = PD_one_hold_out_Dataset(root=root + '/' + dataset_factory, state='test',
                                           transform=data_transform,
                                           )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=4)

    net.eval()
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path,map_location='cpu'), strict=False)
    if len(missing_keys) and len(unexpected_keys) != 0:
        print(missing_keys, unexpected_keys)

    save_cam_per = '../checkpoints/cam/' + version + '_' + '/layer4'
    if not os.path.exists(save_cam_per):
        os.makedirs(save_cam_per)

    num = 0
    for step, data in enumerate(test_loader):
        img, label, name = data[0], data[1], data[2]


        # ---------------------- #
        # show Grad-CAM
        # ---------------------- #
        target_layer = net.layer4[-1]
        input_tensor = img.cuda()

        # init GradCAM 核心代码
        cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)
        target_category = label.tolist()
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        # 可视化
        for i in range(batch_size):

            grayscale_cam_ = grayscale_cam[i, :]

            rgb_img = img_ten2arr(img[i])

            # ① 可视化 原图叠加热图
            visualization = show_cam_on_image(img=rgb_img / 255, mask=grayscale_cam_, use_rgb=True)
            plt.cla()
            plt.close('all')
            plt.figure()
            # plt.imshow(visualization)
            plt.imshow(visualization)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            name_i = os.path.split(name[i])[1]
            plt.savefig(save_cam_per + '/' + name_i.replace(".svg", '_gt' + str(label[i].item()) + '_num'+ str(num) + '.png'),
                        format="svg", bbox_inches='tight', pad_inches=0)
            num+=1
            plt.show()
            plt.close()
            plt.clf()  # 用于批量存储图片时 每一次显示图片并保存以后，释放图窗，接受下一个图片显示和存储


            # # ② 可视化 原图、原图叠加热图
            # visualization = show_cam_on_image(img=rgb_img / 255, mask=grayscale_cam_, use_rgb=True)
            # plt.cla()
            # plt.close('all')
            # plt.figure()
            # # plt.imshow(visualization)
            # plt.imshow(np.concatenate([rgb_img,visualization], axis=1))
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            #
            # name_i = os.path.split(name[i])[1]
            # # plt.savefig(save_cam_per + '/' + name_i.replace(".svg", '_gt' + str(label[i].item()) + '.png'),
            # #             format="svg", bbox_inches='tight', pad_inches=0)
            # plt.show()
            # plt.close()
            # plt.clf()
            # break

        break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 参考链接：https://github.com/jacobgil/pytorch-grad-cam
    # 参数初始化
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    setup_seed(42)
    root = '/disk1/imed_hjq/data/University/Parkinsonism'
    dataset_factory = 'oriR2_split_fileR1'  
    num_classes = 2
    version = 'oh_20230602R1_oriR2_split_fileR1_wavesrnet' 
    model_weight_path = '../checkpoints/save_weights/' + version + '/best_model.pt'

    model_type = 'wavesrnet'

    batch_size = 2
    img_resize = 224
    img_crop = 224

    print('model_weight_path:', model_weight_path)
    print('num_classes:', num_classes)

    main(root, dataset_factory, model_type, version, model_weight_path, batch_size, num_classes, img_resize,
         img_crop)


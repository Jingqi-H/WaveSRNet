"""
参考链接：https://github.com/shap/shap
综述论文：https://www.sciencedirect.com/science/article/pii/S1361841522001177#bib0012

https://github.com/inouye-lab/ShapleyExplanationNetworks
Shapley Explanation Networks：被设计用来嵌入到网络中的解释，理论上可以用于训练好的CNN网络，但是得自己设计

本代码参考链接：
https://blog.paperspace.com/deep-learning-model-interpretability-with-shap/
其他参考资料：https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
"""

import torch
import sys
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
from sklearn.metrics import roc_curve, confusion_matrix, cohen_kappa_score
import pandas as pd
import random
from torch.utils.data import random_split
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')
import shap

sys.path.append('/disk1/imed_hjq/code/3-PD_analysis/02_baseline/')
from dataloaders.pd_dataset import PD_one_hold_out_Dataset
import torch.utils.data as DATA
from utils.metrics import metrics_score_binary, cm_metric
from train_utils.init_model import init_model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def plot_shap(image_array, mask, model):
    """
    This function performs model explainability
    by producing shap plots for a data instance
    """
    image_tensor = image_array.detach().cuda()
    # -----------------
    #  CLASSIFICATION
    # -----------------
    #  creating a mapping of classes to labels
    label_dict = {0: 'HC', 1: 'PD'}

    #  utilizing the model for classification
    with torch.no_grad():
        outs = model(image_tensor)
        prediction = torch.argmax(outs, dim=1).cpu().numpy().tolist()
        print('pre:', prediction)

    #  displaying model classification
    # print(f'prediction: {label_dict[prediction]}')

    # ----------------
    #  EXPLANABILITY
    # ----------------
    #  creating dataloader for mask
    # mask_loader = DataLoader(mask, batch_size=64)

    #  creating explainer for model behaviour
    for train_data in mask:
        images = train_data[0].cuda()
        # print(images.shape)
        explainer = shap.DeepExplainer(model, images)
        # break

    # 一张一张绘制
    for idx, image in enumerate(image_tensor):
        #  deriving shap values for image of interest based on model behaviour
        shap_values = explainer.shap_values(image.view(-1, 1, 224, 224))

        #  preparing for visualization by changing channel arrangement
        shap_numpy = [np.swapaxes(np.swapaxes(x, 1, -1), 1, 2) for x in shap_values]
        image_numpy = np.swapaxes(np.swapaxes(image.view(-1, 1, 224, 224).cpu().numpy(), 1, -1), 1, 2)

        #  producing shap plots
        shap.image_plot(shap_numpy, image_numpy, show=False, labels=['HC', 'PD'])  # show=Falsec才能保存，show=True只展示，保存为空
        plt.savefig('../checkpoints/plot/shap/gt{}pre{}_{}'.format(label[idx], prediction[idx],
                                                                   Path(name[idx]).with_suffix(".png")), format="png",
                    bbox_inches='tight', pad_inches=0.1)
        plt.savefig('../checkpoints/plot/shap/gt{}pre{}_{}'.format(label[idx], prediction[idx],
                                                                   Path(name[idx]).with_suffix(".svg")), format="svg",
                    bbox_inches='tight', pad_inches=0.1)
        plt.savefig('../checkpoints/plot/shap/gt{}pre{}_{}'.format(label[idx], prediction[idx],
                                                                   Path(name[idx]).with_suffix(".pdf")), format="pdf",
                    bbox_inches='tight', pad_inches=0.1)
        # break


#  defining dataset class
class CustomMask(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transforms != None:
            image = self.transforms(image)
        return image


batch_size = 432
# class_num = 10
# dim = 32
root = '/disk1/imed_hjq/data/WenzhouMedicalUniversity/Parkinsonism'
dataset_factory = 'oriR2_split_fileR1'
# version = 'oh_20230410R1_oriR2_split_fileR1_res18'
# model_type = 'res18'
version = 'oh_20230602R1_oriR2_split_fileR1_ours7back'
model_type = 'ours7back'
model_weight_path = '../checkpoints/save_weights/' + version + '/best_model.pt'

net = init_model(model_type, 2, 'cuda')
net.cuda()
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
if len(missing_keys) and len(unexpected_keys) != 0:
    print('load weight error.')
    print('missing_keys:{}\nunexpected_keys:{}'.format(missing_keys, unexpected_keys))

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = PD_one_hold_out_Dataset(root=root + '/' + dataset_factory, state='train',
                                        transform=data_transform,
                                        )
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=4)
test_dataset = PD_one_hold_out_Dataset(root=root + '/' + dataset_factory, state='test',
                                       transform=data_transform,
                                       )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=4)

for i, data in enumerate(test_loader):
    img, label, name = data[0], data[1], data[2]
    print('gt:', label)
    plot_shap(img, train_loader, net)
    break

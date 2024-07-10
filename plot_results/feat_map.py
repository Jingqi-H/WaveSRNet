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
import random
from torch.utils.data import random_split
import numpy as np
import os
from torchvision import transforms
import sys

sys.path.append('/disk1/imed_hjq/code/3-PD_analysis/02_baseline/')
from dataloaders.pd_dataset import PD_one_hold_out_Dataset
from train_utils.init_model import init_model
from utils.metrics import cm_metric

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

"""
测试集中所有图片，画四个频率的特征图，没有用
"""


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init(model_type, num_classes,img_resize,img_crop, path, bs, public_data=False):
    net = init_model(model_type, num_classes, 'cuda')
    net.cuda()

    data_transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop((img_crop, img_crop)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = PD_one_hold_out_Dataset(root=path, state='test',
                                           transform=data_transform,
                                           public_data=public_data)
    print('the num of testing set: {}.'.format(len(test_dataset)))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=bs,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=4)

    return net, test_loader


def get_feat(layer=4, index=4):
    fmap_block = list()
    def farward_hook(module, input, output):
        fmap_block.append(input[0].data)

    net.eval()
    with torch.no_grad():
        labels, pred_prob_all = [], []
        for step, [img, label, name] in enumerate(test_loader):
            hook = eval("net.layer{}[{}].waveatt.register_forward_hook(farward_hook)".format(layer, index - 1))

            outputs = net(img.cuda())
            hook.remove()
            # labels+=label.cpu().data.numpy().tolist()
            labels.append(label)

            logits_softmax = torch.softmax(outputs, dim=1)
            pred_prob_all.append(logits_softmax.cpu().detach())

    weights = torch.cat(fmap_block, 0).squeeze().cpu().data # shape:[432, 64, 56, 56]
    print(weights.shape)

    indices_0 = torch.nonzero(torch.concat(labels,dim=0) == 0).squeeze()
    indices_1 = torch.nonzero(torch.concat(labels,dim=0) == 1).squeeze()
    # indices_2 = torch.nonzero(torch.concat(labels,dim=0) == 2).squeeze()

    weights_0 = weights.index_select(0, indices_0) # torch.Size([306, 64, 56, 56])
    weights_1 = weights.index_select(0, indices_1) # torch.Size([126, 64, 56, 56])

    # 只绘制了一张oct图的前12张特征图
    im = np.transpose(weights_0[0].numpy(), [1, 2, 0])
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()
    plt.close()



if __name__ == '__main__':
    setup_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    root = '/disk1/imed_hjq/data/WenzhouMedicalUniversity/Parkinsonism'
    # root = '/disk1/imed_hjq/data/OCT_AMD'
    dataset_factory = 'oriR2_split_fileR1'  #
    # dataset_factory = 'OCT_AMD_R1'
    num_classes = 2
    # num_classes = 3
    version = 'oh_20230602R1_oriR2_split_fileR1_ours7back'
    # version = 'oh_20230602R1_OCT_AMD_R1_ours7back'
    model_weight_path = '../checkpoints/save_weights/' + version + '/best_model.pt'
    model_type = 'ours7back'

    batch_size = 2
    img_resize = 224
    img_crop = 224

    print('model_weight_path:', model_weight_path)
    print('num_classes:', num_classes)

    public_data = False
    if dataset_factory != 'oriR2_split_fileR1':
        public_data = True

    net, test_loader = init(model_type, num_classes,img_resize,img_crop, root + '/' + dataset_factory, batch_size, public_data)

    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    if len(missing_keys) and len(unexpected_keys) != 0:
        print(missing_keys, unexpected_keys)

    # print(net)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    get_feat(1, 2)
    # w3_2 = get_feat(2, 2)
    # w4_2 = get_feat(3, 2)
    # w5_2 = get_feat(4, 2)
    # # print('{}\n{}.'.format(w5_2[0],w5_2[1]))
    # np.savez('Data2_feat_vec_mean.npz', w2_2=w2_2, w3_2=w3_2, w4_2=w4_2, w5_2=w5_2)


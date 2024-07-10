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

sys.path.append('/data1/hjq/project/3-PD_analysis/02_WaveSRNet/')
from dataloaders.pd_dataset import PD_one_hold_out_Dataset
from train_utils.init_model import init_model
from utils.metrics import cm_metric

# import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')

"""
测试集中所有图片，画特征的折线图：分成两个类别，绘制不同频率系数，绘制不同深度的特征向量的均值
论文中没用
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

    test_dataset = PD_one_hold_out_Dataset(root=root + '/' + dataset_factory, state='test',
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
        fmap_block.append(output.data)

    net.eval()
    with torch.no_grad():
        labels, pred_prob_all = [], []
        for step, [img, label, name] in enumerate(test_loader):
            hook = eval(
                # "net.layer{}[{}].waveatt.cSE.sigmoid.register_forward_hook(farward_hook)".format(layer, index - 1))
            "net.layer{}[{}].waveatt.fc[2].register_forward_hook(farward_hook)".format(layer, index - 1))
            # "net.layer{}[{}].waveatt.fc.3.register_forward_hook(farward_hook)".format(layer, index - 1))

            outputs = net(img.cuda())
            hook.remove()
            # labels+=label.cpu().data.numpy().tolist()
            labels.append(label)

            logits_softmax = torch.softmax(outputs, dim=1)
            pred_prob_all.append(logits_softmax.cpu().detach())

    weights = torch.cat(fmap_block, 0).squeeze().cpu().data # shape:(432, 64)

    indices_0 = torch.nonzero(torch.concat(labels,dim=0) == 0).squeeze()
    indices_1 = torch.nonzero(torch.concat(labels,dim=0) == 1).squeeze()
    # indices_2 = torch.nonzero(torch.concat(labels,dim=0) == 2).squeeze()

    weights_0 = weights.index_select(0, indices_0)
    weights_1 = weights.index_select(0, indices_1)
    # weights_2 = weights.index_select(0, indices_2)

    weight0 = np.mean(weights_0.numpy(), 0)
    weight1 = np.mean(weights_1.numpy(), 0)
    # weight2 = np.mean(weights_2.numpy(), 0)
    # return [weight0, weight1, weight2]
    return [weight0, weight1]



if __name__ == '__main__':
    setup_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    root = '/data1/hjq/data/WenzhouMedicalUniversity/Parkinsonism/oriR2'
    dataset_factory = 'disk12data1_oriR2_split_fileR1'

    num_classes = 2
    version = 'oh_20230602R1_oriR2_split_fileR1_ours7back'
    model_weight_path = '../checkpoints/save_weights/PD_sota/' + version + '/best_model.pt'
    model_type = 'wavesrnet'

    batch_size = 2
    img_resize = 224
    img_crop = 224

    print('model_weight_path:', model_weight_path)
    print('num_classes:', num_classes)

    public_data = False
    if dataset_factory != 'disk12data1_oriR2_split_fileR1':
        public_data = True

    net, test_loader = init(model_type, num_classes,img_resize,img_crop, root + '/' + dataset_factory, batch_size, public_data)

    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    if len(missing_keys) and len(unexpected_keys) != 0:
        print(missing_keys, unexpected_keys)

    # print(net)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)


    # w2_2 = get_feat(1, 2)
    # w3_2 = get_feat(2, 2)
    # w4_2 = get_feat(3, 2)
    # w5_2 = get_feat(4, 2)
    # np.savez('Data1_feat_vec_mean.npz', w2_2=w2_2, w3_2=w3_2, w4_2=w4_2, w5_2=w5_2)

    #  # 师兄的,画折线图
    # data = np.load('./Data1_feat_vec_mean.npz')
    # w2_2, w3_2, w4_2, w5_2 = data['w2_2'], data['w3_2'], data['w4_2'], data['w5_2']
    # n = 16
    # for i in [w2_2, w3_2, w4_2, w5_2]:
    #     for id, weights_all in enumerate(i):
    #         a = [t[0] for t in np.split(np.arange(weights_all.shape[0]), n)]
    #         weight_all = np.vstack([t.mean(0) for t in np.hsplit(weights_all, n)]).squeeze()
    #         plt.plot(a, weight_all, marker="o", label=str(id))
    #     plt.xlabel('channel index')
    #     plt.ylabel('activation')
    #     plt.legend()
    #     plt.show()

    '总结：0类和1类的特征向量在选择后的没有任何区分度'

    # jing
    data = np.load('Data1_feat_vec_mean.npz')
    w2_2, w3_2, w4_2, w5_2 = data['w2_2'], data['w3_2'], data['w4_2'], data['w5_2']
    # data = {"0": w2_2[0], "1": w2_2[1], "2": w2_2[2]}
    data = {"0": w2_2[0], "1": w2_2[1]}
    df = pd.DataFrame(data)
    sns.lineplot(data=df, palette='Set1', linewidth=2, linestyle='-')
    plt.show()
    plt.close()
    #
    # data = {"0": w3_2[0], "1": w3_2[1], "2": w3_2[2]}
    data = {"0": w3_2[0], "1": w3_2[1]}
    df = pd.DataFrame(data)
    sns.lineplot(data=df, palette='Set1', linewidth=2, linestyle='-')
    plt.show()
    plt.close()
    #
    # data = {"0": w4_2[0], "1": w4_2[1], "2": w4_2[2]}
    data = {"0": w4_2[0], "1": w4_2[1]}
    df = pd.DataFrame(data)
    sns.lineplot(data=df, palette='Set1', linewidth=2, linestyle='-')
    plt.show()
    plt.close()
    #
    # data = {"0": w5_2[0], "1": w5_2[1], "2": w5_2[2]}
    data = {"0": w5_2[0], "1": w5_2[1]}
    df = pd.DataFrame(data)
    sns.lineplot(data=df, palette='Set1', linewidth=2, linestyle='-')
    plt.show()
    plt.close()


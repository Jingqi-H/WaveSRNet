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
sns.set_style("whitegrid")

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
测试集中所有图片，获得不同频率系数的重要性分数，绘制fig9
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
    out = list()
    scores = list()
    before, after, coe = list(), list(), list()
    def farward_hook(module, input, output):
        out.append(output[0].data)
        scores.append(output[1].data)
        before.append(output[2].data)
        after.append(output[3].data)
        coe.append(output[4].data)

    net.eval()
    with torch.no_grad():
        labels, pred_prob_all, names = [], [], []
        for step, [img, label, name] in enumerate(test_loader):
            hook = eval(
            "net.layer{}[{}].waveatt.dct_layer.register_forward_hook(farward_hook)".format(layer, index - 1))

            outputs = net(img.cuda())
            hook.remove()
            # labels+=label.cpu().data.numpy().tolist()
            labels.append(label)
            names+=list(name)

            logits_softmax = torch.softmax(outputs, dim=1)
            pred_prob_all.append(logits_softmax.cpu().detach())

    outs = torch.cat(out, 0).squeeze().cpu().data # shape:[432, 64]
    scores = torch.cat(scores, 0).squeeze().cpu().data # shape:
    feats_before = torch.cat(before, 0).squeeze().cpu().data # shape:
    feats_after = torch.cat(after, 0).squeeze().cpu().data # shape:[432, 4, 64]
    coe = torch.cat(coe, 1).squeeze().cpu().data #[4, 432, 64, 28, 28]

    print('outs:',outs.shape)
    print('feats_after:',feats_after.shape)
    print('coe:',coe.shape)

    data = {
        "scores": scores,
        "feats_before": feats_before,
        "feats_after": feats_after,
        "labels": torch.concat(labels,dim=0),
        "names": names,
        "coe": coe,
    }
    np.save('feat_freq_S_{}_{}.npy'.format(layer, index), data)


if __name__ == '__main__':


    setup_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    """
    1 保存特征选择前后的特征
    """
    # root = '/data1/hjq/data/WenzhouMedicalUniversity/Parkinsonism/oriR2'
    # # root = '/disk1/imed_hjq/data/OCT_AMD'
    # dataset_factory = 'disk12data1_oriR2_split_fileR1'  #
    # # dataset_factory = 'OCT_AMD_R1'
    # num_classes = 2
    # # num_classes = 3
    # version = 'oh_20230602R1_oriR2_split_fileR1_ours7back'
    # model_weight_path = '../checkpoints/save_weights/PD_sota/' + version + '/best_model.pt'
    # model_type = 'wavesrnet'
    #
    # batch_size = 2
    # img_resize = 224
    # img_crop = 224
    #
    # print('model_weight_path:', model_weight_path)
    # print('num_classes:', num_classes)
    #
    # public_data = False
    # if dataset_factory != 'disk12data1_oriR2_split_fileR1':
    #     public_data = True
    #
    # net, test_loader = init(model_type, num_classes,img_resize,img_crop, root + '/' + dataset_factory, batch_size, public_data)
    #
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # if len(missing_keys) and len(unexpected_keys) != 0:
    #     print(missing_keys, unexpected_keys)
    #
    # # print(net)
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #
    # get_feat(1, 2)
    # get_feat(2, 2)
    # get_feat(3, 2)
    # get_feat(4, 2)
    # # breakpoint()

    """
    2 加载数据，开始绘图，绘制两类图：每张图片的，各个阶段的四个频率的选择前后特征kde；绘制所有测试集选择前后的均值kde
    修改file的路径，可查看不同阶段的特征分布
    """
    file = 'feat_freq_S_4_2'
    data = np.load(file + '.npy',allow_pickle=True)
    data = data.item()
    labels = data['labels']
    scores = data['scores']
    names = data['names']  #list， 其他是tensor
    feats_before = data['feats_before'] # 得到的四个频率，均值之后的
    feats_after = data['feats_after']
    coe = data['coe']  # 进去选择之前的四格频率特征，测试集样本数432：[4, 432, 64, 28, 28]
    print('feats_before:{}'.format(feats_before.shape))
    print('feats_after:{}'.format(feats_after.shape))
    print('coe:{}'.format(coe.shape))

    # print(scores_0.shape) # [306, 4]
    # print(scores_1.shape) #[126, 4]
    # print(feats_before_0.shape) # [306, 4, 512]
    # print(feats_after_0.shape) # [306, 4, 512]
    # print(len(names_0)) #306

    '2.1 可视化选择前后的四个频率系数特征图，选择前用的是LL, LH, HL, HH = self.dwt_custom1(input)'
    '选择后用的是feats_after，也就是y_after = F.relu(y_before - Q)'
    lenged = ["LL", "LH", "HL", "HH"]
    sns.set_palette("deep") # Set1
    colors = sns.color_palette(n_colors=4)
    #
    # 2.1.1 选择前的
    print('before selection:')
    save_dir = '../checkpoints/plot/2024.6.26selction/{}_before'.format(file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for id in range(coe.shape[1]):  # 第id张测试图
        if id !=0 and labels[id].item() ==0:
            continue
        if id !=308 and labels[id].item() ==1:
            continue
        im_id = coe[:, id, :, :, :]
        for co_id in range(coe.shape[0]):  # 第co_id个频率系数，共四个
            im = np.transpose(im_id[co_id, :, :, :].numpy(), [1, 2, 0])
            features0 = torch.tensor(im.flatten().reshape(im.shape[2], -1))
            magnitudes = torch.mean(features0, dim=1)
            normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
            ax = sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[co_id]}', color=colors[co_id])
            ax.legend(fontsize=15, loc='upper right')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_ylabel('Density', fontsize=20)
            # ax.set_ylim(0,4.5)

        plt.savefig(save_dir + '/gt{}_num{}.svg'.format(labels[id].item(), id), format="svg",
                                        bbox_inches='tight', pad_inches=0.1)
        # plt.show()
        plt.close()
        # break

    # 2.1.2 选择后的
    print('after selection:')
    save_dir = '../checkpoints/plot/2024.6.26selction/{}_after'.format(file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for id in range(feats_after.shape[0]):  # 第id张测试图
        if id != 0 and labels[id].item() == 0:
            continue
        if id != 308 and labels[id].item() == 1:
            continue
        for co_id in range(feats_after.shape[1]):  # 第co_id个频率系数，共四个
            magnitudes = feats_after[id, co_id, :]
            normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
            ax = sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[co_id]}', color=colors[co_id])
            ax.legend(fontsize=15, loc='upper right')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_ylabel('Density', fontsize=20)
            # ax.set_ylim(0,4.5)

        plt.savefig(save_dir + '/gt{}_num{}.svg'.format(labels[id].item(), id), format="svg",
                    bbox_inches='tight', pad_inches=0.1)
        # plt.show()
        plt.close()
        # break


    "2.2 取平均"
    print('calculate mean beafore:')
    save_dir = '../checkpoints/plot/2024.6.26selction_mean'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im_id = torch.mean(coe,dim=1)
    for co_id in range(coe.shape[0]):  # 第co_id个频率系数，共四个
        im = np.transpose(im_id[co_id, :, :, :].numpy(), [1, 2, 0])
        features0 = torch.tensor(im.flatten().reshape(im.shape[2], -1))
        magnitudes = torch.mean(features0, dim=1)
        normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
        ax = sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[co_id]}', color=colors[co_id])
        ax.legend(fontsize=15, loc='upper right')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel('Density', fontsize=20)
        # ax.set_ylim(0, 4.5)

    plt.savefig('{}/{}_before.svg'.format(save_dir,file), format="svg",
                                    bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()
    # break

    print('calculate mean after:')
    save_dir = '../checkpoints/plot/2024.6.26selction_mean'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for co_id in range(feats_after.shape[1]):  # 第co_id个频率系数，共四个
        magnitudes = torch.mean(feats_after, dim=0)[co_id]
        normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
        ax = sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[co_id]}', color=colors[co_id])
        ax.legend(fontsize=15, loc='upper right')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel('Density', fontsize=20)
        # ax.set_ylim(0, 4.5)

    plt.savefig('{}/{}_after.svg'.format(save_dir,file), format="svg",
                bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()
    # break

    '一个图画四个频率系数的数，分别是选择前和选择后的，每个样本都画，共有样本数*2张图'
    # save_dir = '../checkpoints/plot/a_selction_{}'.format(file)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # lenged = ["LL", "LH", "HL", "HH"]
    # save = ['bef', 'aft']
    # for id, feat in enumerate([feats_before, feats_after]):
    #     for num in range(feat.shape[0]):
    #         a_np = feat[num]
    #         for i in range(4):
    #             # magnitudes = a_np[i]
    #             # normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    #             sns.kdeplot(a_np[i], label=f'Feature {lenged[i]}')
    #             # sns.lineplot(data=a_np[i], palette='Set1', linewidth=2, linestyle='-')
    #
    #         # 设置图形标题和坐标轴标签
    #         # plt.title('Kernel Density Estimation for 4 Feature Vectors')
    #         # plt.xlabel('Value')
    #         plt.ylabel('Density')
    #         # 添加图例
    #         if id == 0:
    #             plt.legend()
    #         # 显示图形
    #         plt.savefig(save_dir + '/{}{}_num{}.png'.format(save[id],labels[num].item(), num), format="png",
    #                     bbox_inches='tight', pad_inches=0.1)
    #         # plt.show()
    #         plt.close()
    #         # break


    '所有样本取平均，一个图画四个频率系数的数，共两个图，分别是选择前和选择后的'
    # save_dir = '../checkpoints/plot/all_sample_mean_selction_{}'.format(file)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # lenged = ["LL", "LH", "HL", "HH"]
    # save = ['bef', 'aft']
    # for id, feat in enumerate([feats_before, feats_after]):
    #     a_np = torch.mean(feat, dim=0)
    #     for i in range(4):
    #         magnitudes = a_np[i]
    #         normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    #         sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[i]}')
    #         # sns.lineplot(data=a_np[i], palette='Set1', linewidth=2, linestyle='-')
    #
    #     plt.ylabel('Density')
    #     # 添加图例
    #     if id == 0:
    #         plt.legend()
    #     plt.savefig(save_dir + '/{}.png'.format(save[id]), format="png",
    #                 bbox_inches='tight', pad_inches=0.1)
    #     plt.show()
    #     plt.close()

    '将数据分成0和1类'
    # indices_0 = torch.nonzero(labels == 0).squeeze()
    # indices_1 = torch.nonzero(labels == 1).squeeze()
    # # indices_2 = torch.nonzero(torch.concat(labels,dim=0) == 2).squeeze()
    #
    # scores_0 = scores.index_select(0, indices_0)
    # scores_1 = scores.index_select(0, indices_1)
    # feats_before_0 = feats_before.index_select(0, indices_0)
    # feats_before_1 = feats_before.index_select(0, indices_1)
    # feats_after_0 = feats_after.index_select(0, indices_0)
    # feats_after_1 = feats_after.index_select(0, indices_1)
    # names_0 = [names[i] for i in indices_0.tolist()]
    # names_1 = [names[i] for i in indices_1.tolist()]
    # save_dir = '../checkpoints/plot/all_sample_mean_cls_selction_{}'.format(file)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save = ["LL", "LH", "HL", "HH"]
    # # lenged = ['before selection', 'after selection']
    # lenged = ['Control', 'PD']
    # for i in range(4):
    #     feat1 = feats_before_0[:,i,:]
    #     feat2 = feats_before_1[:,i,:]
    #     for id, feat in enumerate([feat1, feat2]):
    #         magnitudes = torch.mean(feat, dim=0)
    #         normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    #         sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[id]}')
    #         # sns.lineplot(data=normalized_magnitudes, palette='Set1', linewidth=2, label=f'Feature {lenged[id]}')
    #     plt.legend()
    #     plt.savefig(save_dir + '/before_feat_{}.png'.format(save[i]), format="png",
    #                 bbox_inches='tight', pad_inches=0.1)
    #     plt.show()
    #     plt.close()
    #
    # for i in range(4):
    #     feat1 = feats_after_0[:,i,:]
    #     feat2 = feats_after_1[:,i,:]
    #     for id, feat in enumerate([feat1, feat2]):
    #         magnitudes = torch.mean(feat, dim=0)
    #         normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    #         sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[id]}')
    #         # sns.lineplot(data=normalized_magnitudes, palette='Set1', linewidth=2, label=f'Feature {lenged[id]}')
    #     plt.legend()
    #     plt.savefig(save_dir + '/after_feat_{}.png'.format(save[i]), format="png",
    #                 bbox_inches='tight', pad_inches=0.1)
    #     plt.show()
    #     plt.close()


    # print(scores_0.shape) # [306, 4]
    # print(scores_1.shape) #[126, 4]
    # print(feats_before_0.shape) # [306, 4, 512]
    # print(feats_after_0.shape) # [306, 4, 512]
    # print(len(names_0)) #306
    #
    # save_dir = '../checkpoints/plot/selction_{}'.format(file)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # lenged = ["LL", "LH", "HL", "HH"]
    # save = ['bef0', 'bef1', 'aft0', 'aft1']
    # for id, feat in enumerate([feats_before_0,feats_before_1, feats_after_0, feats_after_1]):
    #     # if save[id] != 'bef0':
    #     #     continue
    #     for num in range(feat.shape[0]):
    #         a_np = feat[num]
    #         # a_np = feat[num].numpy()
    #         for i in range(4):
    #             magnitudes = a_np[i]
    #             normalized_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    #             sns.kdeplot(normalized_magnitudes, label=f'Feature {lenged[i]}')
    #         # 设置图形标题和坐标轴标签
    #         # plt.title('Kernel Density Estimation for 4 Feature Vectors')
    #         # plt.xlabel('Value')
    #         plt.ylabel('Density')
    #         # 添加图例
    #         if id == 0:
    #             plt.legend()
    #         # 显示图形
    #         plt.savefig(save_dir + '/{}_num{}.png'.format(save[id], num), format="png",
    #                     bbox_inches='tight', pad_inches=0.1)
    #         # plt.show()
    #         plt.close()

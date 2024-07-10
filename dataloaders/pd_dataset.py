import os

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as DATA
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import random



class PD_one_hold_out_Dataset(Dataset):
    def __init__(self, root, transform=None, state='train', public_data=False):
        super(PD_one_hold_out_Dataset, self).__init__()
        self.info_path = root + '.csv'
        self.state = state
        self.transform = transform
        self.public_data = public_data

        self.info = pd.read_csv(self.info_path)
        self.info_file = self.info[self.state].dropna().values.tolist()


    def __getitem__(self, index):
        path = self.info_file[index]

        if self.public_data:
            label = int(path.split('/')[-2][-1])
        elif self.info_path.split('/')[-1] in ['PD2_cls.csv', 'PD2_cls_into2.csv']:
            label = int(path.split('/')[-4])
        else:
            label = int(path.split('/')[-3])

        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split('/')[-1]

    def __len__(self):
        return len(self.info_file)

class PD_Kfold(Dataset):
    """
        为了获得batchsize的每个图片名字
    """

    def __init__(self, root, transform=None, state='train', k=0, public_data=False):
        super(PD_Kfold, self).__init__()
        self.info_path = '{}/oriR2_to_K{}.csv'.format(root, str(k))
        self.state = state
        self.transform = transform
        self.public_data = public_data

        self.info = pd.read_csv(self.info_path)
        self.info_file = self.info[self.state].dropna().values.tolist()


    def __getitem__(self, index):
        path = self.info_file[index]

        if self.public_data:
            label = int(path.split('/')[-2][-1])
        elif self.info_path.split('/')[-1] in ['PD2_cls.csv', 'PD2_cls_into2.csv']:
            label = int(path.split('/')[-4])
        else:
            label = int(path.split('/')[-3])

        # https://blog.csdn.net/qq_30017409/article/details/121400373
        # 单通道转多通道的原理实际上是将单通道的图像复制3份，在显示结果上不存在差异
        img = Image.open(path).convert("L")
        # img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split('/')[-1]  # path.split("\\")[-1]

    def __len__(self):
        return len(self.info_file)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    num_epoch = 1
    seed = 42
    k_fold = 5

    setup_seed(seed)

    data_path_train = '/data1/hjq/data/WenzhouMedicalUniversity/Parkinsonism/oriR2/oriR2_split_fileR1'
    data_transform = transforms.Compose([
        transforms.RandomRotation(10,expand=True),
        transforms.Resize((232)),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

    ])

    dataset = PD_one_hold_out_Dataset(data_path_train, state='val', transform=data_transform)
    print(len(dataset))

    dataloader = DATA.DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=4)

    for epoch in range(num_epoch):
        for step, data in enumerate(dataloader, start=0):
            images, labels, names = data
            # print(transforms.ToPILImage()(images[0]))
            plt.imshow(transforms.ToPILImage()(images[0]))
            plt.show()
            print(images.shape)
            # print(images[:,:,20:50,20:50])
            # print(images[:,:,128:135,128:135].shape)
            # print(images[:,:,128:135,128:135])
            print(labels)
            # print(names)
            # break

            if step == 10:
                break

        break

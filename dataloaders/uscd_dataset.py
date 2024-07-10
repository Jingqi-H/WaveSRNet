import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
import torch.utils.data as DATA
from skimage import io, img_as_ubyte
import numpy as np
import cv2
from PIL import Image



class USCDdataset(ImageFolder):

    def __init__(self, root, transform=None):
        super(USCDdataset, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        label = self.imgs[index][1]

        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split("/")[-1]


if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt

    """
    修改了数据集每个类别文件夹的名称，在原始名字之前加上类别索引，正常的是0
    """

    root = '/disk1/imed_hjq/data/USCD'
    batch_size = 64

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # AddPepperNoise(0.98, p=0.5),
            # transforms.RandomRotation(degrees=(5, 10)),
            transforms.Resize((232)),  # ori:  2048, 2730
            transforms.RandomCrop([224, 224]),
            # transforms.Resize((224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(5, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((250)),  # ori:  2048, 2730
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),

        "test": transforms.Compose([
            transforms.Resize((224)),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    dataset = USCDdataset(root=root+'/training', transform=data_transform['val'])

    cls_list = dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cls_list.items())
    cla = []
    for key, val in cls_list.items():
        cla.append(key)
    # write dict into json file
    print(cla_dict)
    json_str = json.dumps(cla_dict, indent=4)
    with open('uscd_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print(len(dataset))

    dataloader = DATA.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)

    for epoch in range(1):
        for step, data in enumerate(dataloader, start=0):
            images, labels, names = data
            # print(transforms.ToPILImage()(images[0]))
            plt.imshow(transforms.ToPILImage()(images[0]),cmap='Greys_r')
            plt.show()
            print(images.shape)
            # print(images[:,:,20:50,20:50])
            # print(images[:,:,128:135,128:135].shape)
            # print(images[:,:,128:135,128:135])
            print(labels)
            break






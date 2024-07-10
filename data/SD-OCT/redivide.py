import os
import cv2
import random
import shutil


in_path = '/data/home/xiaozunjie/datasets/Publication_Dataset'
train_out_path = '/data/home/zhangxiaoqing/Zhangxq/OCT_AMD/train'
test_out_path = '/data/home/zhangxiaoqing/Zhangxq/OCT_AMD/test'

if not os.path.isdir(train_out_path):
    os.makedirs(train_out_path)
if not os.path.isdir(test_out_path):
    os.makedirs(test_out_path)

dataset = os.listdir(in_path)
AMD_dataset = []  # 2
DME_dataset = []  # 1
NORMAL_dataset = []  # 0

for i in dataset:
    if i[0] == 'A':
        AMD_dataset.append(i)
    elif i[0] == 'D':
        DME_dataset.append(i)
    else:
        NORMAL_dataset.append(i)

random.shuffle(AMD_dataset)
random.shuffle(DME_dataset)
random.shuffle(NORMAL_dataset)

train_set = []
test_set = []

# 8:2
for i in AMD_dataset[:int(len(AMD_dataset) * 0.8)]:
    train_set.append(i)
for i in AMD_dataset[int(len(AMD_dataset) * 0.8):]:
    test_set.append(i)

for i in DME_dataset[:int(len(DME_dataset) * 0.8)]:
    train_set.append(i)
for i in DME_dataset[int(len(DME_dataset) * 0.8):]:
    test_set.append(i)

for i in NORMAL_dataset[:int(len(NORMAL_dataset) * 0.8)]:
    train_set.append(i)
for i in NORMAL_dataset[int(len(NORMAL_dataset) * 0.8):]:
    test_set.append(i)

for i in train_set:
    target_path = train_out_path + '/' + i

    if i[0] == 'A':
        target_path = target_path + '_2'
    elif i[0] == 'D':
        target_path = target_path + '_1'
    else:
        target_path = target_path + '_0'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for root, dirs, files in os.walk(in_path + '/' + i + '/TIFFs/8bitTIFFs'):
        for file in files:
            src_file = cv2.imread(os.path.join(root, file))
            cv2.imwrite(os.path.join(target_path, file[:-4] + '.png'), src_file)

for i in test_set:
    target_path = test_out_path + '/' + i

    if i[0] == 'A':
        target_path = target_path + '_2'
    elif i[0] == 'D':
        target_path = target_path + '_1'
    else:
        target_path = target_path + '_0'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for root, dirs, files in os.walk(in_path + '/' + i + '/TIFFs/8bitTIFFs'):
        for file in files:
            src_file = cv2.imread(os.path.join(root, file))
            cv2.imwrite(os.path.join(target_path, file[:-4] + '.png'), src_file)


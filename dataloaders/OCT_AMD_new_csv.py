import os
import shutil
import random
import torch
import numpy as np
import csv
import pandas as pd



def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(42)
val_split_rate = 0.2

root = '/data1/hjq/data/OCT_AMD/'


train_list = os.listdir(root+'/train')
test_index = os.listdir(root+'/test')
random.shuffle(train_list)

val_index = random.sample(train_list, k=int(len(train_list) * val_split_rate))
print(test_index,val_index)


train_file, val_file, test_file = [], [], []
for index, case in enumerate(test_index):
    file_list = os.listdir(os.path.join(root+'/test', case))
    for file in file_list:
        img_path = os.path.join(root+'/test', case, file)
        test_file.append(img_path)

for index, case in enumerate(train_list):
    print(case)
    file_list = os.listdir(os.path.join(root+'/train', case))
    if case in val_index:
        for file in file_list:

            img_path = os.path.join(root+'/train', case, file)
            val_file.append(img_path)

    else:
        for file in file_list:
            img_path = os.path.join(root+'/train', case, file)
            train_file.append(img_path)

print('val_file:',val_file)
print('train_file:',train_file)
print('test_file:',test_file)


s1 = pd.DataFrame({'train': train_file})
s2 = pd.DataFrame({'val': val_file})
s3 = pd.DataFrame({'test': test_file})
df = pd.concat((s1, s2, s3), axis=1)
df.fillna('', inplace=True)  # 将NaN替换成空白字符串 # , inplace=True

df.to_csv("./OCT_AMD_R1.csv",index=False) # , index=0
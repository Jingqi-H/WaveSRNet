import torch
from torchvision import transforms
import os
import time, datetime
import numpy as np
import pandas as pd
import random
from torch.utils.data import random_split
import seaborn as sns

from dataloaders.pd_dataset import PD_one_hold_out_Dataset
from utils.metrics import metrics_score_binary, cm_metric
from train_utils.init_model import init_model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(root, dataset_factory, version, model_type, model_weight_path, batch_size, num_classes, img_resize,
         img_crop):

    net = init_model(model_type, num_classes)
    net.cuda()

    # parm = {}
    # for name, parameters in net.named_parameters():
    #     if name == 'Mixed_6e.branch7x7_1.bn.bias':  # vgg:feats.features.0.bias | googlenet:inception5b.branch1.conv.weight
    #         print(name, ':', parameters.size())
    #         print(name, ':', parameters)
    #         parm[name] = parameters.cpu().detach().numpy()

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
    model_dict = torch.load(model_weight_path,map_location='cpu')
    missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)
    print('missing_keys:{}\nunexpected_keys:{}'.format(missing_keys, unexpected_keys))

    # pred_probs, gt_labels = 0, 0
    # val_acc, recall, precision, auc, f1 = 0, 0, 0, 0, 0
    pred_prob_all, gt_label, names = [], [], []
    results = {'img_name': [], 'gt': [], 'pro': []}
    with torch.no_grad():
        # pred_prob, pred_label = 0, 0
        for step, data in enumerate(test_loader):
            img, label, name = data[0], data[1], data[2]

            output = net(img.cuda())

            logits_softmax = torch.softmax(output, dim=1)
            pred_prob = logits_softmax[:, 1].cpu().detach()

            pred_prob_all.append(pred_prob)
            gt_label.append(label)

            results['img_name']+=list(name)
            results['gt']+=label.numpy().tolist()
            results['pro']+=pred_prob.numpy().tolist()


    pred_probs = np.concatenate(pred_prob_all)
    gt_labels = np.concatenate(gt_label)

    # 保存结果
    data_frame = pd.DataFrame(data=results)
    # data_frame.to_csv('./checkpoints/csv_pro/' + model_type + '_' + version + '.csv',
    #                   index_label='index')

    acc, recall, precision, Specificity, auc, f1, kappa = cm_metric(gt_labels, pred_probs, cls_num=1)
    print('acc:{:.4f} | recall:{:.4f} | precision:{:.4f}| specificity:{:.4f} | auc:{:.4f} | f1:{:.4f}| kappa:{:.4f}'.format(acc, recall, precision, Specificity,auc, f1, kappa))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # 参数初始化
    seed = 42
    setup_seed(seed)
    root = '/data1/hjq/data/WenzhouMedicalUniversity/Parkinsonism/oriR2'
    dataset_factory = 'disk12data1_oriR2_split_fileR1'  #disk12data1_oriR2_split_fileR1 oriR2_split_fileR1

    model_type = 'res18'
    version = 'oh_20230410R1_oriR2_split_fileR1_res18'
    model_weight_path = './checkpoints/save_weights/PD_sota/' + version + '/best_model.pt'

    batch_size = 2
    num_classes = 2

    img_resize = 224
    img_crop = 224

    print('model_weight_path:', model_weight_path)
    print('num_classes:', num_classes)

    start_time = time.time()
    print("start time:", datetime.datetime.now())

    main(root, dataset_factory, version, model_type, model_weight_path, batch_size, num_classes, img_resize,
         img_crop)

    print("\nEnd time:", datetime.datetime.now())

import torch
from torchvision import transforms
import os
import time, datetime
import numpy as np
import random
from torch.utils.data import random_split
from utils.metrics import metrics_score_binary, cm_metric
from train_utils.init_model import init_model
from dataloaders.uscd_dataset import USCDdataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(root, dataset_factory, version, model_type, model_weight_path, batch_size, num_classes, img_resize,
         img_crop):

    net = init_model(model_type, num_classes)
    net.cuda()


    data_transform = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop((img_crop, img_crop)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


    test_dataset = USCDdataset(root=root + '/' + dataset_factory + '/validation',
                               transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=4)

    net.eval()
    model_dict = torch.load(model_weight_path,map_location='cpu')
    missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)
    print('missing_keys:{}\nunexpected_keys:{}'.format(missing_keys, unexpected_keys))

    pred_prob_all, gt_label, names = [], [], []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            img, label, name = data[0], data[1], data[2]

            output = net(img.cuda())

            logits_softmax = torch.softmax(output, dim=1)

            pred_prob_all.append(logits_softmax.cpu().detach())
            gt_label.append(label)


    pred_probs = np.concatenate(pred_prob_all)
    gt_labels = np.concatenate(gt_label)

    # print(gt_labels, '\n',pred_probs.shape)
    acc, recall, precision, Specificity, auc, f1, kappa = cm_metric(gt_labels, pred_probs, cls_num=pred_probs.shape[1])
    print('acc:{:.4f} | recall:{:.4f} | precision:{:.4f}| specificity:{:.4f} | auc:{:.4f} | f1:{:.4f}| kappa:{:.4f}'.format(acc, recall, precision, Specificity,auc, f1, kappa))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    # 参数初始化
    seed = 42
    setup_seed(seed)
    root = '/data1/hjq/data'
    dataset_factory = 'USCD'

    model_type = 'res18'
    version = 'oh_20230615R1_USCD_res18'
    model_weight_path = './checkpoints/save_weights/UCSD_sota/' + version + '/best_model.pt'

    batch_size = 2
    num_classes = 4

    # img_resize = 232
    img_resize = 224
    img_crop = 224

    print('model_weight_path:', model_weight_path)
    print('num_classes:', num_classes)

    start_time = time.time()
    print("start time:", datetime.datetime.now())

    main(root, dataset_factory, version, model_type, model_weight_path, batch_size, num_classes, img_resize,
         img_crop)

    print("\nEnd time:", datetime.datetime.now())

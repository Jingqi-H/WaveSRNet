import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import datetime
from torchvision import transforms
import torchvision.models
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import time
# from pytorchtools import EarlyStopping
import torch.multiprocessing
# from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import math
import torch.optim.lr_scheduler as lr_scheduler

torch.multiprocessing.set_sharing_strategy('file_system')

from train_utils.train_and_eval import train_one_epoch, evaluate
from train_utils.loss_factory import Focal_Loss
from train_utils.early_stop import EarlyStopping
from dataloaders.pd_dataset import PD_one_hold_out_Dataset
from dataloaders.uscd_dataset import USCDdataset
from train_utils.init_model import init_model
from train_utils.loss_factory import CrossEntropyLabelSmooth


def main(args):
    # start_time = time.time()
    # 用来保存训练以及验证过程中信息
    # data_path = data-ori_548
    # results_file = "./checkpoints/logs/resnet_{}_{}_results{}.txt".format(args.data_path, args.wavelength,
    #                                                                       datetime.datetime.now().strftime(
    #                                                                           "%Y%m%d-%H%M%S"))
    results_file = "./checkpoints/logs/oh_{}_{}#{}results.txt".format(args.dataset_factory, args.modelname,
                                                                      args.version)

    print("\nstart time:", datetime.datetime.now())
    setup_seed(args.seed)
    # device = torch.device("cuda:" + args.gpu_ids if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    with open(results_file, "a") as f:
        f.write("start time: " + str(datetime.datetime.now()) + "\n")
        f.write("using {} device.".format(device) + "\n")
        f.write("set seed: " + str(args.seed) + "\n")
        f.write("args: \n" + str(args) + "\n\n")

    print('init full_dataset.')
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((232)),  # ori:  2048, 2730
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
        "val": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(5, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((250)),  # ori:  2048, 2730
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),

        "test": transforms.Compose([
            transforms.Resize((224)),
            transforms.CenterCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    #
    if args.dataset_factory == 'octmnist.npz' and args.public_data:
        file = np.load(args.root + '/' + args.dataset_factory)

        new_dim_input = True
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(file['train_images']).float(),
                                                       torch.from_numpy(file['train_labels']).squeeze().long())
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(file['val_images']).float(),
                                                     torch.from_numpy(file['val_labels']).squeeze().long())
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(file['test_images']).float(),
                                                      torch.from_numpy(file['test_labels']).squeeze().long())

    elif args.dataset_factory == 'USCD':
        new_dim_input = False
        train_dataset = USCDdataset(root=args.root + '/' + args.dataset_factory + '/training',
                                    transform=data_transform["train"])
        val_dataset = USCDdataset(root=args.root + '/' + args.dataset_factory + '/validation',
                                  transform=data_transform["val"])
        test_dataset = USCDdataset(root=args.root + '/' + args.dataset_factory + '/validation',
                                   transform=data_transform["test"])

    else:
        new_dim_input = False
        train_dataset = PD_one_hold_out_Dataset(root=args.root + '/' + args.dataset_factory, state='train',
                                                transform=data_transform["train"],
                                                public_data=args.public_data)  # PDMILDataset
        val_dataset = PD_one_hold_out_Dataset(root=args.root + '/' + args.dataset_factory, state='val',
                                              transform=data_transform["test"], public_data=args.public_data,
                                              )
        test_dataset = PD_one_hold_out_Dataset(root=args.root + '/' + args.dataset_factory, state='test',
                                               transform=data_transform["test"], public_data=args.public_data,
                                               )

    if args.modelname in ['ours7', 'ours7_res50', 'ours7back', 'ours7back50', 'ours7back_srm2layer']:  # 因为模型的原因，val和test需要>1
        drop_last = True
    else:
        drop_last = False
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             drop_last=drop_last,
                                             num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              drop_last=drop_last,
                                              num_workers=4)
    print('len train_dataset:{}\nlen val_dataset:{}\nlen test_dataset:{}'.format(len(train_dataset), len(val_dataset),
                                                                                 len(test_dataset)))
    print('load data end.')

    print('load {} model with pre train {}.'.format(args.modelname, args.pre_train))
    net = init_model(model_name=args.modelname, num_classes=args.num_classes, pre_train=args.pre_train)
    net.to(device)
    print('load model end.')

    # from thop import profile
    # from thop import clever_format
    # macs, params = profile(net, inputs=(torch.randn((2, 1, 224, 224)).to(device),))
    # macs, params = clever_format([macs, params], "%.2f")
    # print('macs:', macs, 'params:', params)
    #
    # from torchstat import stat
    # stat(net.cpu(), (1, 224, 224))

    with open(results_file, "a") as f:
        f.write("Param requires_grad: \n")
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
                f.write(name + "\n")

    if args.loss_factory == 'ce':
        criterion = nn.CrossEntropyLoss().cuda()
        print('init CrossEntropyLoss.')
    elif args.loss_factory == 'ce_label_smooth':
        criterion = CrossEntropyLabelSmooth(num_classes=args.num_classes).cuda()
        print('init CrossEntropyLabelSmooth.')
    else:
        criterion = None
        print('init Loss false.')


    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          dampening=0,  # 动量的抑制因子，默认为0
                          weight_decay=args.weight_decay,  # 默认为0，有值说明用作正则化
                          nesterov=True, )  # 使用Nesterov动量，默认为False
    scheduler = None

    writer_train = SummaryWriter(
        './checkpoints/run_tensorboard/oh_{}_{}_{}/Train'.format(args.version, args.dataset_factory, args.modelname))
    writer_val = SummaryWriter(
        './checkpoints/run_tensorboard/oh_{}_{}_{}/Val'.format(args.version, args.dataset_factory, args.modelname))
    best_ep, best_acc, best_recall, best_precision, best_auc, best_f1 = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(args.num_epochs):

        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        train_scores, train_mean_loss, lr_ = train_one_epoch(net, optimizer, train_loader, device, criterion,
                                                             epoch,
                                                             args.lr,
                                                             args.num_epochs, new_dim_input=new_dim_input,
                                                             scheduler=scheduler
                                                             )
        lr_ = optimizer.param_groups[-1]['lr']

        writer_train.add_scalar('loss/total', train_mean_loss, epoch)
        writer_train.add_scalar('score/acc', train_scores[0], epoch)
        writer_train.add_scalar('score/recall', train_scores[1], epoch)
        writer_train.add_scalar('score/precision', train_scores[2], epoch)
        writer_train.add_scalar('score/specificity', train_scores[3], epoch)
        writer_train.add_scalar('score/auc', train_scores[4], epoch)
        writer_train.add_scalar('score/f1', train_scores[5], epoch)
        writer_train.add_scalar('score/kappa', train_scores[6], epoch)
        writer_train.add_scalar('lr', lr_, epoch)

        # validate
        val_scores, val_mean_loss = evaluate(net, val_loader, device, criterion, new_dim_input=new_dim_input)
        writer_val.add_scalar('loss/total', val_mean_loss, epoch)
        writer_val.add_scalar('score/acc', val_scores[0], epoch)
        writer_val.add_scalar('score/recall', val_scores[1], epoch)
        writer_val.add_scalar('score/precision', val_scores[2], epoch)
        writer_train.add_scalar('score/specificity', val_scores[3], epoch)
        writer_val.add_scalar('score/auc', val_scores[4], epoch)
        writer_val.add_scalar('score/f1', val_scores[5], epoch)
        writer_val.add_scalar('score/kappa', val_scores[6], epoch)

        print('[Epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, train_mean_loss, val_mean_loss))
        print(
            "val acc:{:.4f}, recall:{:.4f}, precision:{:.4f}, specificity:{:.4f}, auc:{:.4f}, f1:{:.4f}, kappa:{:.4f}".format(
                val_scores[0],
                val_scores[1],
                val_scores[2],
                val_scores[3],
                val_scores[4],
                val_scores[5],
                val_scores[6]))

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            # Precision(查准率) and Recall(查全率)
            train_info = f"\n\n[EPOCH: {epoch + 1}]\n" \
                         f"Lr: {lr_:.6f}\n" \
                         f"TRAIN:\n" \
                         f"loss: {train_mean_loss:.4f}\n" \
                         f"acc: {train_scores[0]:.4f}  " \
                         f"recall: {train_scores[1]:.4f}  " \
                         f"precision: {train_scores[2]:.4f}  " \
                         f"specificity: {train_scores[3]:.4f}  " \
                         f"auc: {train_scores[4]:.4f}  " \
                         f"f1: {train_scores[5]:.4f}  " \
                         f"kappa: {train_scores[6]:.4f}  "

            val_info = f"\nEVAL:\n" \
                       f"loss: {val_mean_loss:.4f}\n" \
                       f"acc: {val_scores[0]:.4f}  " \
                       f"recall: {val_scores[1]:.4f}  " \
                       f"precision: {val_scores[2]:.4f}  " \
                       f"specificity: {val_scores[3]:.4f}  " \
                       f"auc: {val_scores[4]:.4f}  " \
                       f"f1: {val_scores[5]:.4f}  " \
                       f"kappa: {val_scores[6]:.4f}  "

            f.write(train_info + val_info)
            # f.write(train_info + "\n\n")

        # break

        if val_scores[0] > best_acc:
            best_ep = epoch + 1
            best_acc, best_recall, best_precision, best_auc, best_f1 = val_scores[0], val_scores[1], val_scores[2], \
                                                                       val_scores[3], val_scores[4]
            with open(results_file, "a") as f:
                f.write("\nEP{} save best!".format(str(best_ep)))

            test_scores, test_mean_loss = evaluate(net, test_loader, device, criterion, new_dim_input=new_dim_input)
            # Accary, Recall, Precision, Specificity, roc_auc, F1
            print(
                "test acc:{:.4f}, recall:{:.4f}, precision:{:.4f}, specificity:{:.4f}, auc:{:.4f}, f1:{:.4f}, kappa:{:.4f}".format(
                    test_scores[0],
                    test_scores[1],
                    test_scores[2],
                    test_scores[3],
                    test_scores[4],
                    test_scores[5],
                    test_scores[6]))

            with open(results_file, "a") as f:
                test_info = f"\nTEST:\n" \
                            f"loss: {test_mean_loss:.4f}\n" \
                            f"acc: {test_scores[0]:.4f}  " \
                            f"recall: {test_scores[1]:.4f}  " \
                            f"precision: {test_scores[2]:.4f}  " \
                            f"specificity: {test_scores[3]:.4f}  " \
                            f"auc: {test_scores[4]:.4f}  " \
                            f"f1: {test_scores[5]:.4f}  " \
                            f"kappa: {test_scores[6]:.4f}"

                f.write(test_info)

        save_model_path = "./checkpoints/save_weights/oh_{}_{}_{}".format(args.version, args.dataset_factory,
                                                                          args.modelname)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        save_model_path = "{}/best_model.pt".format(save_model_path)

        early_stopping(val_scores[0], 'acc', net, save_model_path)

        if early_stopping.early_stop:
            print("Early stopping")
            with open(results_file, "a") as f:
                f.write("\nEarly stopping")
            break

    with open(results_file, "a") as f:
        f.write("\n\n===END!===")



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--root", default='/data1/hjq/data/WenzhouMedicalUniversity/Parkinsonism/oriR2', help="root")
    parser.add_argument('--dataset_factory', default='disk12data1_oriR2_split_fileR1', type=str, help='dataset_factory')
    parser.add_argument('--modelname', default='res18', type=str, help='model')  # inception,transformer,res18,senet
    parser.add_argument('--version', default='today', type=str, help='version')
    parser.add_argument("--num_epochs", default=5, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--patience", default=2, type=int, help="patience")
    parser.add_argument('--gpu_ids', default="1", type=str, help='gpu_ids')
    parser.add_argument('--pre_train', action='store_true', help='if python main.py --pre_train, then true.')
    parser.add_argument('--public_data', action='store_true', help='if train on public_data, please --public_data')

    parser.add_argument('--loss_factory', default='ce', type=str, help='loss_factory')

    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--val_bs", default=1, type=int)

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    # parser.add_argument('--save-best', default=True, type=bool, help='only save best acc weights')
    parser.add_argument('--save_best', action='store_false', help='only save best acc weights')

    args = parser.parse_args()

    return args


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not os.path.exists("checkpoints/save_weights"):
        os.mkdir("checkpoints/save_weights")
    if not os.path.exists("checkpoints/logs"):
        os.mkdir("checkpoints/logs")

    main(args)

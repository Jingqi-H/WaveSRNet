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

torch.multiprocessing.set_sharing_strategy('file_system')

from train_utils.train_and_eval import train_one_epoch, evaluate
from train_utils.loss_factory import Focal_Loss
from train_utils.early_stop import EarlyStopping
from dataloaders.pd_dataset import PD_Kfold
from train_utils.init_model import init_model

"""
 EarlyStopping早停法的实现原理:
 https://blog.csdn.net/qq_35054151/article/details/115986287
"""


def main(args, k, results_file):

    print("\nstart time:", datetime.datetime.now())
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    with open(results_file, "a") as f:
        f.write("start time: " + str(datetime.datetime.now()) + "\n")
        f.write("using {} device.".format(device) + "\n")

    print('split data.')
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((232)),
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

    train_dataset = PD_Kfold(root=args.root + '/' + args.dataset_factory, state='train', k=k,
                             transform=data_transform["train"],
                             public_data=False)  # PDMILDataset
    val_dataset = PD_Kfold(root=args.root + '/' + args.dataset_factory, state='val', k=k,
                           transform=data_transform["test"], public_data=False,
                           )
    test_dataset = PD_Kfold(root=args.root + '/' + args.dataset_factory, state='test', k=k,
                            transform=data_transform["test"], public_data=False,
                            )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=4)
    print('len train_dataset:{}\nlen val_dataset:{}\nlen test_dataset:{}'.format(len(train_dataset), len(val_dataset),
                                                                                 len(test_dataset)))
    print('load data end.')

    print('load {} model with pre train {}.'.format(args.modelname, args.pre_train))
    net = init_model(model_name=args.modelname, num_classes=args.num_classes)
    net.to(device)
    print('load model end.')

    with open(results_file, "a") as f:
        f.write("Param requires_grad: \n")
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
                f.write(name + "\n")

    criterion = nn.CrossEntropyLoss().cuda()
    print('init CrossEntropyLoss.')


    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          dampening=0,
                          weight_decay=args.weight_decay,
                          nesterov=True, )


    writer_train = SummaryWriter(
        './checkpoints/run_tensorboard/{}_{}_{}/Train'.format(args.version, args.dataset_factory, args.modelname))
    writer_val = SummaryWriter(
        './checkpoints/run_tensorboard/{}_{}_{}/Val'.format(args.version, args.dataset_factory, args.modelname))
    writer_test = SummaryWriter(
        './checkpoints/run_tensorboard/{}_{}_{}/Test'.format(args.version, args.dataset_factory, args.modelname))
    best_val_acc, best_val_kappa, best_ep, best_acc, best_recall, best_precision, best_specific, best_auc, best_f1, best_kappa = 0.0,0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(args.num_epochs):

        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        train_scores, train_mean_loss, lr_ = train_one_epoch(net, optimizer, train_loader, device, criterion,
                                                             epoch,
                                                             args.lr,
                                                             args.num_epochs,
                                                             )
        writer_train.add_scalar('loss/total', train_mean_loss, epoch)
        writer_train.add_scalar('score/acc', train_scores[0], epoch)
        writer_train.add_scalar('score/recall', train_scores[1], epoch)
        writer_train.add_scalar('score/precision', train_scores[2], epoch)
        writer_train.add_scalar('score/auc', train_scores[4], epoch)
        writer_train.add_scalar('score/f1', train_scores[5], epoch)
        writer_train.add_scalar('score/kappa', train_scores[6], epoch)
        writer_train.add_scalar('lr', lr_, epoch)

        # validate
        val_scores, val_mean_loss = evaluate(net, val_loader, device, criterion, new_dim_input=False)
        writer_val.add_scalar('loss/total', val_mean_loss, epoch)
        writer_val.add_scalar('score/acc', val_scores[0], epoch)
        writer_val.add_scalar('score/recall', val_scores[1], epoch)
        writer_val.add_scalar('score/precision', val_scores[2], epoch)
        writer_val.add_scalar('score/auc', val_scores[4], epoch)
        writer_val.add_scalar('score/f1', val_scores[5], epoch)
        writer_val.add_scalar('score/kappa', val_scores[6], epoch)


        print('[Epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, train_mean_loss, val_mean_loss))
        print(
            "val acc:{:.4f}, recall:{:.4f}, precision:{:.4f}, spec:{:.4f}, auc:{:.4f}, f1:{:.4f}, kappa:{:.4f}".format(
                val_scores[0],
                val_scores[1],
                val_scores[2],
                val_scores[3],
                val_scores[4],
                val_scores[5],
                val_scores[6], ))

        with open(results_file, "a") as f:
            train_info = f"\n\n[EPOCH: {epoch + 1}]\n" \
                         f"Lr: {lr_:.6f}\n" \
                         f"TRAIN:\n" \
                         f"loss: {train_mean_loss:.4f}\n" \
                         f"acc: {train_scores[0]:.4f}  " \
                         f"recall: {train_scores[1]:.4f}  " \
                         f"precision: {train_scores[2]:.4f}  " \
                         f"spec: {train_scores[3]:.4f}  " \
                         f"auc: {train_scores[4]:.4f}  " \
                         f"f1: {train_scores[5]:.4f}  " \
                         f"kappa: {train_scores[6]:.4f}  "
            val_info = f"\nEVAL:\n" \
                       f"loss: {val_mean_loss:.4f}\n" \
                       f"acc: {val_scores[0]:.4f}  " \
                       f"recall: {val_scores[1]:.4f}  " \
                       f"precision: {val_scores[2]:.4f}  " \
                       f"spec: {val_scores[3]:.4f}" \
                       f"auc: {val_scores[4]:.4f}" \
                       f"f1: {val_scores[5]:.4f}" \
                       f"kappa: {val_scores[6]:.4f}  "

            f.write(train_info + val_info)


        if val_scores[0] > best_val_acc:
            best_val_acc = val_scores[0]

            best_ep = epoch + 1

            with open(results_file, "a") as f:
                f.write("\nEP{} save best!".format(str(best_ep)))

            test_scores, test_mean_loss = evaluate(net, test_loader, device, criterion, new_dim_input=False)
            # Accary, Recall, Precision, Specificity, roc_auc, F1
            print(
                "test acc:{:.4f}, recall:{:.4f}, precision:{:.4f}, specificity:{:.4f}, auc:{:.4f}, f1:{:.4f}, kappa:{:.4f}".format(
                    test_scores[0],
                    test_scores[1],
                    test_scores[2],
                    test_scores[3],
                    test_scores[4],
                    test_scores[5],
                    test_scores[6], ))

            best_acc, best_recall, best_precision, best_specific, best_auc, best_f1, best_kappa = test_scores[0], \
                                                                                                  test_scores[1], \
                                                                                                  test_scores[2], \
                                                                                                  test_scores[3], \
                                                                                                  test_scores[4], \
                                                                                                  test_scores[5], \
                                                                                                  test_scores[6],
            with open(results_file, "a") as f:
                test_info = f"\nTEST:\n" \
                            f"loss: {test_mean_loss:.4f}\n" \
                            f"acc: {test_scores[0]:.4f}  " \
                            f"recall: {test_scores[1]:.4f}  " \
                            f"precision: {test_scores[2]:.4f}  " \
                            f"specificity: {test_scores[3]:.4f}  " \
                            f"auc: {test_scores[4]:.4f}   " \
                            f"f1: {test_scores[5]:.4f}   " \
                            f"kappa: {test_scores[6]:.4f}"

                f.write(test_info)

        save_model_path = "./checkpoints/save_weights/{}_{}_{}".format(args.version, args.dataset_factory,
                                                                       args.modelname)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        save_model_path = "{}/k{}_best_model.pt".format(save_model_path, str(k))

        early_stopping(val_scores[0], 'acc', net, save_model_path)

        if early_stopping.early_stop:
            print("Early stopping")
            with open(results_file, "a") as f:
                f.write("\nEarly stopping")
            break

    with open(results_file, "a") as f:
        f.write("\n\n===END!===\n\n")
    # k fold 删除过去的
    # https://github.com/DLLXW/data-science-competition/blob/03490a7ea8e6297211fe8709b61ddad251ae51dd/kaggle/Cassava%20Leaf%20Disease%20Classification/train.py#L454
    del net, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    return best_ep, [best_acc, best_recall, best_precision, best_specific, best_auc, best_f1, best_kappa]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--root", default='/data1/hjq/project/3-PD_analysis/02_baseline/rebuttal_code', help="root")
    parser.add_argument('--dataset_factory', default='data_kfold', type=str, help='dataset_factory')
    parser.add_argument('--modelname', default='res18', type=str, help='model')
    parser.add_argument('--version', default='today', type=str, help='version')
    parser.add_argument("--num_epochs", default=5, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--patience", default=2, type=int, help="patience")
    parser.add_argument('--gpu_ids', default="7", type=str, help='gpu_ids')
    parser.add_argument('--pre_train', action='store_true',
                        help='if python main_kf_rebuttal.py --pre_train, then true.')

    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch_size", default=64, type=int)

    # patience
    parser.add_argument("--kfold", default=5, type=int, help="patience")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
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
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not os.path.exists("checkpoints/save_weights"):
        os.mkdir("checkpoints/save_weights")
    if not os.path.exists("checkpoints/logs"):
        os.mkdir("checkpoints/logs")

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    results_file = "./checkpoints/logs/kfold_{}#{}results.txt".format(args.modelname, args.version)
    with open(results_file, "w") as f:
        f.write("set seed: " + str(args.seed) + "\n")
        f.write("args: \n" + str(args) + "\n\n")

    ep, acc, recall, precision, spec, auc, f1, kappa = [], [], [], [], [], [], [], []
    for k in range(args.kfold):
        print('\n\n=====KFold [{}/{}]====='.format(k + 1, args.kfold))
        with open(results_file, "a") as f:
            f.write('=====KFold [{}/{}]=====\n'.format(k + 1, args.kfold))

        best_ep, best_score = main(args, k, results_file)
        ep.append(best_ep)
        acc.append(best_score[0])
        recall.append(best_score[1])
        precision.append(best_score[2])
        spec.append(best_score[3])
        auc.append(best_score[4])
        f1.append(best_score[5])
        kappa.append(best_score[6])
        # break

    print('\n')
    print('EP:{}'.format(ep))
    print('acc:{}\nrecall:{}\nprecision:{}\nspecificity:{}\nauc:{}\nf1:{}\nkappa:{}\n'.format(acc, recall, precision,
                                                                                              spec, auc, f1, kappa))
    print('acc Mean {:.4f}'.format(np.mean(acc)))
    print('acc STD {:.4f}'.format(np.std(acc)))
    print('recall Mean {:.4f}'.format(np.mean(recall)))
    print('recall STD {:.4f}'.format(np.std(recall)))
    print('precision Mean {:.4f}'.format(np.mean(precision)))
    print('precision STD {:.4f}'.format(np.std(precision)))
    print('Specificity Mean {:.4f}'.format(np.mean(spec)))
    print('Specificity STD {:.4f}'.format(np.std(spec)))
    print('auc Mean {:.4f}'.format(np.mean(auc)))
    print('auc STD {:.4f}'.format(np.std(auc)))
    print('f1 Mean {:.4f}'.format(np.mean(f1)))
    print('f1 STD {:.4f}'.format(np.std(f1)))
    print('kappa Mean {:.4f}'.format(np.mean(kappa)))
    print('kappa STD {:.4f}'.format(np.std(kappa)))

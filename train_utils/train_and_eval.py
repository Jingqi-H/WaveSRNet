import torch
import numpy as np

from utils.metrics import metrics_score_binary, cm_metric


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, base_lr, total_niters, lr_power):
    lr = lr_poly(base_lr, i_iter, total_niters, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_epoch(optimizer, epoch,base_learning_rate):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9:
        # warm-up training for large minibatch
        lr = base_learning_rate + base_learning_rate* epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def evaluate(model, data_loader, device, criterion, is_mil=False, modelname='model',new_dim_input=False):
    model.eval()
    running_loss = []
    pred_prob_all, label_all = [], []
    with torch.no_grad():
        for val_steps, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]

            if new_dim_input:
                imgs = imgs.unsqueeze(dim=1)

            logits = model(imgs.to(device))
            logits_softmax = torch.softmax(logits, dim=1)
            pred_prob_all.append(logits_softmax.cpu().detach())
            label_all.append(labels)

            loss = criterion(logits, labels.long().to(device))
            running_loss.append(loss.item())

    pred_probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(label_all)

    if pred_probs.shape[1] == 2:
        val_score = cm_metric(labels, pred_probs[:,1], cls_num=1)
    else:
        val_score = cm_metric(labels, pred_probs, cls_num=pred_probs.shape[1])

    return val_score, np.mean(running_loss)


def train_one_epoch(model, optimizer, data_loader, device, criterion,
                    per_epoch,
                    base_lr,
                    max_epoch,is_mil=False, modelname='model',new_dim_input=False,scheduler=None):
    model.train()
    total_niters = max_epoch * len(data_loader)
    optimizer.zero_grad()

    running_loss = []
    pred_prob_all, label_all = [], []
    lr_list = []
    # adjust_lr = 0
    for train_steps, data in enumerate(data_loader, start=0):

        imgs, labels = data[0], data[1]

        current_idx = per_epoch * len(data_loader) + train_steps
        adjust_lr = adjust_learning_rate(optimizer, current_idx, base_lr=base_lr,
                                         total_niters=total_niters,
                                         lr_power=0.9)

        if new_dim_input:
            imgs = imgs.unsqueeze(dim=1)
        logits = model(imgs.to(device))

        logits_softmax = torch.softmax(logits, dim=1)
        pred_prob_all.append(logits_softmax.cpu().detach())

        label_all.append(labels)
        loss = criterion(logits, labels.long().to(device))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if not scheduler is None:
            scheduler.step()

        running_loss.append(loss.item())


    pred_probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(label_all)

    if pred_probs.shape[1] == 2:
        train_score = cm_metric(labels, pred_probs[:,1], cls_num=1)
    else:
        train_score = cm_metric(labels, pred_probs, cls_num=pred_probs.shape[1])
    return train_score, np.mean(running_loss), adjust_lr
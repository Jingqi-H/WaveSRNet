import torch
import torch.nn as nn


class Focal_Loss(torch.nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_v1(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # import pdb; pdb.set_trace()
        loss = (- targets * log_probs).mean(0).sum()
        return loss

    def forward_v2(self, inputs, targets):
        probs = self.logsoftmax(inputs)
        targets = torch.zeros(probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = nn.KLDivLoss()(probs, targets)
        return loss

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        return self.forward_v1(inputs, targets)

    def test(self):
        inputs = torch.randn(2, 5)
        targets = torch.randint(0, 5, [2])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        print((targets * torch.log(targets) - targets * log_probs).sum(-1).mean())
        print(nn.KLDivLoss(reduce='mean')(log_probs, targets))
        # import pdb; pdb.set_trace()



if __name__ == '__main__':
    a = torch.rand([4])
    print(a)
    b = torch.tensor([1,1,0,0])
    fa_loss = Focal_Loss()
    loss = fa_loss(a, b)
    print(loss)

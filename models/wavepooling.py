import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavepool_utils.pool_select import get_pool


"""
https://github.com/Fraunhofer-SCAI/wavelet_pooling/blob/27b9e1e80b/mnist_pool.py
"""


class Net(nn.Module):
    def __init__(self, pool_type):
        super(Net, self).__init__()
        self.pool_type = pool_type
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, stride=1)
        self.norm1 = nn.BatchNorm2d(20)
        self.pool1 = get_pool(self.pool_type, scales=3)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=0, stride=1)
        self.norm2 = nn.BatchNorm2d(50)
        self.pool2 = get_pool(self.pool_type, scales=2)
        self.conv3 = nn.Conv2d(50, 500, 4, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(1)
        if self.pool_type == 'adaptive_wavelet':
            self.lin = nn.Linear(500, 10)
        else:
            self.lin = nn.Linear(500, 10)
        self.norm4 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.norm1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin(x)
        # x = self.norm4(x)
        output = F.log_softp(x, dim=1)
        return output

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
                or self.pool_type == 'scaled_adaptive_wavelet':
            return self.pool1.wavelet.wavelet_loss() + \
                   self.pool2.wavelet.wavelet_loss()
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1, self.pool2]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.pool1.wavelet, self.pool2.wavelet]
        else:
            return []


class LeNet5(nn.Module):
    def __init__(self, pool_type, num_classes):
        super(LeNet5, self).__init__()
        self.pool_type = pool_type
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, padding=2, stride=1)
        self.act1 = nn.ReLU()
        self.s2 = get_pool(pool_type, scales=2, out_shape=(14, 14))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=5, padding=0, stride=1)
        self.act3 = nn.ReLU()
        self.s4 = get_pool(pool_type, scales=2, out_shape=(5, 5))
        self.fc1 = nn.Linear(46656, 120)  # 46656  16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.fc =nn.Sequential(
                nn.Linear(46656, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(64, num_classes),
            )

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.c1(x)
        x = self.act1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.act3(x)
        x = self.s4(x)
        x = x.reshape(b, -1)
        # 20230601
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        output = self.fc(x)
        return output

    def get_wavelet_loss(self):
        if self.pool_type == 'adaptive_wavelet'\
                or self.pool_type == 'scaled_adaptive_wavelet':
            return self.s2.wavelet.wavelet_loss() + \
                   self.s4.wavelet.wavelet_loss()
        else:
            return torch.tensor(0.)

    def get_pool(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.s2, self.s4]
        else:
            return []

    def get_wavelets(self):
        if self.pool_type == 'adaptive_wavelet' \
           or self.pool_type == 'scaled_adaptive_wavelet':
            return [self.s2.wavelet, self.s4.wavelet]
        else:
            return []


if __name__ == '__main__':
    pooling_type = 'adaptive_wavelet'
    net = LeNet5(pool_type=pooling_type, num_classes=2)
    print(net)
    net.cuda()

    temp = torch.randn((16, 1, 224, 224)).cuda()
    y = net(temp)
    print(y.shape)
import torch
import torch.nn as nn

# Migration from https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/models/wide_residual_network.py
class WideResNet(nn.Module):
    def __init__(self, in_channels, num_classes=(4, 3, 3), depth=16, widen_factor=4, drop_rate=0.3):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_channels[0],
                               kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = ConvGroup(n_channels[0], n_channels[1], n, 1, drop_rate).cuda()
        self.conv3 = ConvGroup(n_channels[1], n_channels[2], n, 2, drop_rate).cuda()
        self.conv4 = ConvGroup(n_channels[2], n_channels[3], n, 2, drop_rate).cuda()

        self.bn = nn.BatchNorm2d(n_channels[3]).cuda()
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=8)

        assert len(num_classes) == 3
        self.fc_rot = nn.Linear(n_channels[3], num_classes[0])
        self.fc_h = nn.Linear(n_channels[3], num_classes[1])
        self.fc_v = nn.Linear(n_channels[3], num_classes[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_rot(x), self.fc_v(x), self.fc_h(x)


class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, n, strides, drop_rate=0.0):
        super(ConvGroup, self).__init__()

        self.first_block = BasicBlock(in_channels, out_channels, strides, drop_rate)

        self.rest_blocks = []
        for i in range(1, n):
            self.rest_blocks.append(
                BasicBlock(out_channels, out_channels, 1, drop_rate)
            )
        self.n = n

    def forward(self, x):
        x = self.first_block(x)
        for i in range(0, self.n - 1):
            x = self.rest_blocks[i](x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels).cuda()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=strides, padding=1).cuda()
        self.bn2 = nn.BatchNorm2d(num_features=out_channels).cuda()
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout2d(p=drop_rate)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=strides, padding=1).cuda()

        self.is_channels_equal = (in_channels == out_channels)
        if not self.is_channels_equal:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=strides, padding=0).cuda()

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop(out)
        shortcut = x if self.is_channels_equal else self.conv3(x)
        out = out + shortcut
        return out


if __name__ == '__main__':
    x = torch.ones([16, 3, 32, 32])

    net = WideResNet(3)
    y = net(x)

    print(y)

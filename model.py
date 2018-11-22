import math

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        up_blk_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.blk2 = ResBlk(64)
        self.blk3 = ResBlk(64)
        self.blk4 = ResBlk(64)
        self.blk5 = ResBlk(64)
        self.blk6 = ResBlk(64)
        self.blk7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.PReLU())
        blk8 = [UpBlk(64, 2) for _ in range(up_blk_num)]
        blk8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.blk8 = nn.Sequential(*blk8)

    def forward(self, x):
        blk1 = self.blk1(x)
        blk2 = self.blk2(blk1)
        blk3 = self.blk3(blk2)
        blk4 = self.blk4(blk3)
        blk5 = self.blk5(blk4)
        blk6 = self.blk6(blk5)
        blk7 = self.blk7(blk6)
        blk8 = self.blk8(blk1 + blk7)

        return (torch.tanh(blk8) + 1) / 2


class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        b_size = x.size(0)
        return torch.sigmoid(self.net(x).view(b_size))


class ResBlk(nn.Module):
    def __init__(self, channels):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)

        return x + res


class UpBlk(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

import torch
import torch.nn as nn
from torch import Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
        super(DoubleConv, self).__init__()
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-5,
                           momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-5,
                           momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_x2(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U_Net, self).__init__()
        self.encoder1 = DoubleConv(in_channels=in_channels, out_channels=32)
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder2 = DoubleConv(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder3 = DoubleConv(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder4 = DoubleConv(in_channels=128, out_channels=256)
        self.pool4 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bottleneck = DoubleConv(in_channels=256, out_channels=512)
        self.upconv4 = UpConv(512, 256)
        self.decoder4 = DoubleConv(in_channels=512, out_channels=256)
        self.upconv3 = UpConv(256, 128)
        self.decoder3 = DoubleConv(in_channels=256, out_channels=128)
        self.upconv2 = UpConv(128, 64)
        self.decoder2 = DoubleConv(in_channels=128, out_channels=64)
        self.upconv1 = UpConv(64, 32)
        self.decoder1 = DoubleConv(in_channels=64, out_channels=32)
        self.conv = nn.Conv2d(
            32, out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        L1 = self.encoder1(x)
        L2 = self.encoder2(self.pool1(L1))
        L3 = self.encoder3(self.pool2(L2))
        L4 = self.encoder4(self.pool3(L3))

        bottleneck = self.bottleneck(self.pool4(L4))

        R4 = self.upconv4(bottleneck)
        R4 = torch.cat((R4, L4), dim=1)
        R4 = self.decoder4(R4)

        R3 = self.upconv3(R4)
        R3 = torch.cat((R3, L3), dim=1)
        R3 = self.decoder3(R3)

        R2 = self.upconv2(R3)
        R2 = torch.cat((R2, L2), dim=1)
        R2 = self.decoder2(R2)

        R1 = self.upconv1(R2)
        R1 = torch.cat((R1, L1), dim=1)
        R1 = self.decoder1(R1)

        x = self.conv(R1)
        return torch.sigmoid(x)


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, smooth=1) -> Tensor:
        N = target.size(0)

        inputs = input.view(N, -1)
        targets = target.view(N, -1)

        intersection = inputs * targets
        dice = 2.0 * (intersection.sum(1) + smooth) / \
            (inputs.sum(1) + targets.sum(1) + smooth)
        dice_loss = 1 - dice.sum()/N
        return dice_loss

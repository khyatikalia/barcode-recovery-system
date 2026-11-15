"""Small U-Net for barcode image restoration (1 channel in, 1 channel out)."""
import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Input H and W must be divisible by 16 (we use 128 x 384)."""

    def __init__(self, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 8]
        self.e1, self.e2, self.e3 = block(1, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.e4, self.e5 = block(c[2], c[3]), block(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.u4, self.d4 = nn.ConvTranspose2d(c[4], c[3], 2, 2), block(c[3] * 2, c[3])
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[2] * 2, c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[1] * 2, c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        x5 = self.e5(self.pool(x4))
        y = self.d4(torch.cat([self.u4(x5), x4], 1))
        y = self.d3(torch.cat([self.u3(y), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return torch.sigmoid(self.out(y))

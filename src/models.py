import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, normalize, activation_func):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else None
        self.af = activation_func

    def forward(self, x):
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        y = self.af(y)
        return y


class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  padding, normalize, activation_func):
        super(Deconv2dBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else None
        self.af = activation_func

    def forward(self, x):
        y = self.deconv(x)
        if self.bn:
            y = self.bn(y)
        y = self.af(y)
        return y


class Generator(nn.Module):
    def __init__(self, blocks):
        super(Generator, self).__init__()
        self.net = nn.Sequential(*[
            Deconv2dBlock(
                b['in_channels'],
                b['out_channels'],
                b['kernel_size'],
                b['stride'],
                b['padding'],
                b['normalize'],
                b['activation_func']
            ) for b in blocks
        ])

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, blocks):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(*[
            Conv2dBlock(
                b['in_channels'],
                b['out_channels'],
                b['kernel_size'],
                b['stride'],
                b['padding'],
                b['normalize'],
                b['activation_func']
            ) for b in blocks
        ])

    def forward(self, x):
        return self.net(x)

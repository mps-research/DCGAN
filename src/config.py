import torch.nn as nn
from ray import tune


large_netG = {
    'blocks': [
        {
            'in_channels': 100,
            'out_channels': 1024,
            'kernel_size': 4,
            'stride': 1,
            'padding': 0,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 1024,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 512,
            'out_channels': 256,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 256,
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 128,
            'out_channels': 3,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': False,
            'activation_func': nn.Tanh()
        },
    ]
}


large_netD = {
    'p': 0.5,
    'blocks': [
        {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': False,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 512,
            'out_channels': 1,
            'kernel_size': 4,
            'stride': 1,
            'padding': 0,
            'normalize': False,
            'activation_func': nn.Sigmoid()
        },
    ]
}


netGs = {
    'large_netG': large_netG
}


netDs = {
    'large_netD': large_netD
}


config = {
    'netG': tune.grid_search(list(netGs.keys())),
    'netD': tune.grid_search(list(netDs.keys())),
    'lrG': tune.grid_search([2e-4, 2e-5]),
    'lrD': tune.grid_search([2e-4, 2e-5]),
    'batch_size': 128,
    'n_epochs': 10,
}

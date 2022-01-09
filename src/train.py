from ray import tune
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import transforms
from models import Generator, Discriminator, weights_init
from config import config, netGs, netDs


class Trainable(tune.Trainable):
    def setup(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = ImageFolder(root='/data', transform=transform)
        self.dataloader = DataLoader(dataset, self.config['batch_size'], True)

        self.fixed_noises = torch.randn((64, 100, 1, 1), device=self.device)

        self.netG = Generator(**netGs[config['netG']]).to(self.device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(**netDs[config['netD']]).to(self.device)
        self.netD.apply(weights_init)

        self.optimizerG = Adam(self.netG.parameters(), lr=config['lrG'], betas=(0.5, 0.999))
        self.optimizerD = Adam(self.netD.parameters(), lr=config['lrD'], betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

        log_name = self.trial_id + '--' + \
            '--'.join(f'{key}-{value}' for key, value in config.items())
        self.writer = SummaryWriter(f'/logs/{log_name}')

        self.n_updates = 1

    def step(self):
        self.netG.train()
        self.netD.train()
        for data in self.dataloader:
            self.netD.zero_grad()

            images = data[0].to(self.device)
            batch_size = images.size(0)
            labels = torch.full((batch_size, ), 1., dtype=torch.float, device=self.device)
            outputs = self.netD(images).view(-1)
            errD_real = self.criterion(outputs, labels)
            errD_real.backward()

            noises = torch.randn((batch_size, 100, 1, 1), device=self.device)
            images = self.netG(noises)
            labels.fill_(0.)
            outputs = self.netD(images.detach()).view(-1)
            errD_fake = self.criterion(outputs, labels)
            errD_fake.backward()

            errD = (errD_real + errD_fake) / 2.

            self.optimizerD.step()

            self.writer.add_scalar('Discriminator Loss', errD.item(), self.n_updates)

            self.netG.zero_grad()

            labels.fill_(1.)
            outputs = self.netD(images).view(-1)
            errG = self.criterion(outputs, labels)
            errG.backward()

            self.optimizerG.step()

            self.writer.add_scalar('Generator Loss', errG.item(), self.n_updates)

            self.n_updates += 1
        self.netG.eval()
        self.netD.eval()

        images = self.netG(self.fixed_noises)
        image_grid = make_grid(images, normalize=True, value_range=(-1, 1))
        self.writer.add_image('Generated Images', image_grid, self.iteration)

        return {'errG': errG.item(), 'errD': errD.item()}


if __name__ == '__main__':
    tune.run(
        Trainable,
        stop={'training_iteration': 200},
        config=config,
        resources_per_trial={'gpu': 0.2, 'cpu': 0.5}
    )

#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ResNetBlock(nn.Module):
    def __init__(self, n=784):
        super().__init__()

        self.block = nn.Sequential(nn.Tanh(), nn.Linear(n, n, bias=False))

    def forward(self, x):
        return x + self.block(x)


class DenseBlock(nn.Module):
    def __init__(self, n_in=784, n_out=784):
        super().__init__()
        self.block = nn.Sequential(
            nn.Tanh(),
            nn.Linear(n_in, n_out, bias=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNet(nn.Module):
    def __init__(self, n_blocks=1):
        super().__init__()

        layers = [nn.Flatten()]
        for _ in range(n_blocks):
            layers.append(ResNetBlock())

        layers.append(nn.Linear(28 * 28, 10))

        self.resnet_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_stack(x)


class ResNetTower(pl.LightningModule):
    def __init__(self, n_nodes_initials, n_layers_final):
        super().__init__()
        layers = [nn.Flatten()]
        n_nodes_initials = [784] + n_nodes_initials
        for (a, b) in zip(n_nodes_initials[:-1], n_nodes_initials[1:]):
            layers.append(DenseBlock(a, b))

        n = n_nodes_initials[-1]
        for _ in range(n_layers_final):
            layers.append(ResNetBlock(n))

        layers.append(nn.Linear(n, 10))

        self.resnet_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_stack(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x)
        loss = F.nll_loss(pred, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        loss = F.nll_loss(pred, y)
        return loss


if __name__ == "__main__":
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize(0, 255)]),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Normalize(0, 255)]),
    )

    training_dataloader = DataLoader(
        training_data,
        batch_size=64,
        pin_memory=True,
        num_workers=12
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        pin_memory=True,
        num_workers=12
    )

    n_nodes_initials = [512, 256, 128, 64]
    n_layers_final = 128
    n_epochs = 100

    resnet = ResNetTower(n_nodes_initials, n_layers_final)

    trainer = pl.Trainer(precision=16)
    trainer.fit(resnet, training_dataloader, test_dataloader)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Mon Jul 11 20:24:57 2022

Train a ResNet over 100 epochs on the MNIST data set.

@author: ntapiam
"""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from typing import List, Any


class ResNetBlock(nn.Module):
    """Single resnet block.
    :param n: Dimension of the input and output feature space. Assumed to be
        equal
    :type n: int, optional
    """
    def __init__(self, n: int = 784):
        """Constructor method"""
        super().__init__()

        self.block = nn.Sequential(nn.Tanh(), nn.Linear(n, n, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation
        :param x: Input features. Can be batched.
        :type x: torch.Tensor
        :return: Output features
        :rtype: torch.Tensor
        """
        return x + self.block(x)


class DenseBlock(nn.Module):
    """Fully connected NN block.
    :param n_in: Input feature dimension
    :type n_in: int, optional
    :param n_out: Output feature dimension
    :type n_out: int, optional
    """
    def __init__(self, n_in: int = 784, n_out: int = 784):
        """Constructor method"""
        super().__init__()
        self.block = nn.Sequential(
            nn.Tanh(),
            nn.Linear(n_in, n_out, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation
        :param x: Input features. Can be batched
        :type x: torch.Tensor
        """
        return self.block(x)


class ResNet(pl.LightningModule):
    """Full stack of residual blocks forming a ResNet
    :param n_nodes_initials: Number of input features for performing
        dimensional reduction
    :type n_nodes_initials: List[int]
    :param n_layers_final: Number of residual blocks in the stack
    :type n_layers_final: int
    """
    def __init__(self, n_nodes_initials: List[int], n_layers_final: int):
        """Constructor method"""
        super().__init__()
        # Provide example shape for model summary
        self.example_input_array = torch.Tensor(64, 28, 28)

        # Needed to compute model accuracy during validation and testing
        self.val_acc = tm.Accuracy()
        self.test_acc = tm.Accuracy()

        # Start with a Flatten layer to turn input image into vector of pixels
        layers: List[torch.nn.Module] = [nn.Flatten()]

        # Dense blocks to reduce the dimension from 28 * 28
        # down to n_nodes_initials[-1]
        n_nodes_initials = [784] + n_nodes_initials
        for (a, b) in zip(n_nodes_initials[:-1], n_nodes_initials[1:]):
            layers.append(DenseBlock(a, b))

        # Append ResNet blocks to the stack
        n = n_nodes_initials[-1]
        for _ in range(n_layers_final):
            layers.append(ResNetBlock(n))

        layers.append(nn.Linear(n, 10))

        self.resnet_stack = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet
        :param x: Input features. Can be batched
        :type x: torch.Tensor
        :return: Output features
        :rtype: torch.Tensor
        """
        return self.resnet_stack(x)

    def configure_optimizers(self):
        """Configure optimizer for gradien descent"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """Perform one step in training, computing losses to report to
        optimizer
        :param train_batch: Batch of input tensors with labels
        :type train_batch: (torch.Tensor, torch.Tensor)
        :param batch_idx: Index of current batch. Ignored
        :type batch_idx: int
        :return: Loss computed over batch
        :rtype: torch.Tensor
        """
        x, y = train_batch
        pred = self.forward(x)
        loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """Validate the model after every epoch.
        :param val_batch: Batch of input tensors with labels
        :type val_batch: (torch.Tensor, torch.Tensor)
        :param batch_idx: Index of current batch. Ignored
        :type batch_idx: int
        """
        x, y = val_batch
        pred = self.forward(x)
        loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
        self.val_acc(pred, y)
        logs = {"val_loss": loss, "val_acc": self.val_acc}
        self.log_dict(logs)

    def test_step(self, test_batch, batch_idx):
        """Test the model after training.
        :param val_batch: Batch of input tensors with labels
        :type val_batch: (torch.Tensor, torch.Tensor)
        :param batch_idx: Index of current batch. Ignored
        :type batch_idx: int
        """
        x, y = test_batch
        pred = self.forward(x)
        loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
        self.test_acc(pred, y)
        logs = {"test_loss": loss, "test_acc": self.test_acc}
        self.log_dict(logs)


if __name__ == "__main__":
    from torchvision import datasets
    from torch.utils.data import DataLoader, random_split
    from torchvision.transforms import Compose, ToTensor, Normalize
    from multiprocessing import cpu_count

    num_workers = cpu_count()
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

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

    num_samples = int(0.8 * len(training_data))
    train_data, val_data = random_split(
        training_data,
        [num_samples, len(training_data) - num_samples],
        torch.Generator().manual_seed(42)
    )

    training_dataloader = DataLoader(
        train_data,
        batch_size=64,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=64,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers
    )

    n_nodes_initials = [512, 256, 128, 64]
    n_layers_final = 128
    n_epochs = 100

    resnet = ResNet(n_nodes_initials, n_layers_final)

    trainer = pl.Trainer(
        precision=16,
        accelerator=accelerator,
        max_epochs=n_epochs,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(resnet, training_dataloader, val_dataloader)
    trainer.test(resnet, test_dataloader)

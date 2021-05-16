import torch
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(nn.ELU(), nn.Linear(28 * 28, 28 * 28, bias=False))

    def forward(self, x):
        return x + self.block(x)


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

def train_loop(data, model, loss_fn, optim):
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        pred = model(X)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data, model, loss_fn, optim):
    size = len(data.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    training_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    n_blocks = 128
    n_epochs = 100
    learning_rate = 1e-3
    
    resnet = ResNet(n_blocks)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

    for t in range(n_epochs):
        print(f"Epoch {t+1}")
        train_loop(training_dataloader, resnet, loss_fn, optim)
        test_loop(test_dataloader, resnet, loss_fn, optim)

    torch.save(resnet.state_dict(), 'resnet'+n_blocks+'_relu_lin.pth')
    print("Done!")


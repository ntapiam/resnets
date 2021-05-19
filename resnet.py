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


def train_loop(data, model, loss_fn, optim, device):
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optim.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data, model, loss_fn, optim, device):
    size = len(data.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


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
    training_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, pin_memory=True)

    n_blocks = 50
    n_epochs = 100
    learning_rate = 1e-5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    resnet = ResNet(n_blocks).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

    print(f"Training using device: {DEVICE}")
    try:
        for t in range(n_epochs):
            print(f"Epoch {t+1}")
            train_loop(training_dataloader, resnet, loss_fn, optim, DEVICE)
            test_loop(test_dataloader, resnet, loss_fn, optim, DEVICE)

    finally:
        torch.save(resnet.state_dict(), f"resnet{n_blocks}_relu_lin_e{t+1}.pth")
    print("Done!")

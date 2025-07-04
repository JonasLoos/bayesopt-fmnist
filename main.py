from typing import Callable
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
import random


def fix_seed(seed: int = 42) -> None:
    '''
    Fixes the seed for reproducibility.
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fashion_mnist(batch_size: int = 128) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    '''
    Loads the Fashion MNIST dataset.

    Args:
        batch_size (int): The batch size for the dataloader.

    Returns:
        train_loader (torch.utils.data.DataLoader): The dataloader for the training set.
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set.
    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class ResNetLayer(nn.Module):
    '''
    A single layer of the ResNet model.
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        '''
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): The stride of the convolution.
        '''
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the ResNet layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            out (torch.Tensor): The output tensor.
        '''
        # Store input for skip connection
        original_input = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += self.skip_connection(original_input)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    '''
    The ResNet model.
    '''
    def __init__(self, num_classes: int = 10):
        '''
        Args:
            num_classes (int): The number of classes.
        '''
        super().__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Residual layers
        self.layers = nn.Sequential(
            self._make_layer(16, 16, 2, stride=1),
            self._make_layer(16, 32, 2, stride=2),
            # self._make_layer(32, 64, 2, stride=2)
        )

        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        '''
        Makes a layer of the ResNet model.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            num_blocks (int): The number of blocks in the layer.
            stride (int): The stride of the convolution.

        Returns:
            layer (nn.Sequential): The layer of the ResNet model.
        '''
        layers = []
        layers.append(ResNetLayer(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetLayer(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            out (torch.Tensor): The output tensor.
        '''
        # Initial convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Residual layers
        out = self.layers(out)

        # Global average pooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        # Final classification layer
        out = self.fc(out)

        return out


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int = 10, learning_rate: float = 0.001) -> float:
    '''
    Trains the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The dataloader for the training set.
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
    '''
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(f"\rEpoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}", end="", flush=True)

        print()

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            test_loss += criterion(output, target).item()

        test_loss /= len(test_loader)
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {100 * correct / total:.2f}%")

    return test_loss


def sobol_sequence(n: int) -> list[float]:
    '''
    Generates a 1-dimensional Sobol sequence.

    Args:
        n (int): The number of points to generate.

    Returns:
        sequence (list[float]): The Sobol sequence of length n in the interval [0, 1]
    '''

    seq = []
    for i in range(n):
        x, f, k = 0.0, 0.5, i
        while k:
            x += f * (k & 1)   # add reversed bit
            k >>= 1            # shift right
            f *= 0.5           # next fraction
        seq.append(x)
    return seq


def get_next_guess(evaluations: list[tuple[float, float]]) -> float:
    '''
    Gets the next guess using the GP.

    Args:
        evaluations (list[tuple[float, float]]): The evaluations of the function.

    Returns:
        next_guess (float): The next guess.
    '''
    # Convert evaluations to tensors for botorch
    X = torch.tensor([[torch.log10(torch.tensor(lr))] for lr, _ in evaluations], dtype=torch.float64)
    y = torch.tensor([[loss] for _, loss in evaluations], dtype=torch.float64)

    # Standardize y values
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)

    # Initialize GP model (using default parameters)
    gp = SingleTaskGP(X, y)

    # Define acquisition function (Expected Improvement)
    best_f = y.min()
    EI = ExpectedImprovement(gp, best_f=best_f)

    # Optimize acquisition function
    bounds = torch.tensor([[-6.0], [-2.0]], dtype=torch.float64)
    candidate, _ = optimize_acqf(
        EI, bounds=bounds,
        q=1, num_restarts=5, raw_samples=20,
    )
    
    return 10 ** candidate.item()


def bayesian_lr_optimization(model_class: type[nn.Module], train_function: Callable[[nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, float], float], train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int = 10, budget: int = 10) -> float:
    '''
    Bayesian Hyperparameter Optimization for the learning rate.

    Args:
        model_class (type[nn.Module]): The model class to optimize.
        train_loader (torch.utils.data.DataLoader): The dataloader for the training set.
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set.
        epochs (int): The number of epochs to train for.
        budget (int): The number of function evaluations.

    Returns:
        best_lr (float): The best learning rate.
    '''

    evaluations = []
    f = lambda lr: train_function(model_class(), train_loader, test_loader, epochs, lr)

    min_lr_log = -6
    max_lr_log = -2

    # Get initial guesses using Sobol sequence
    sobol_seq = sobol_sequence(min(3, budget))
    initial_guesses = [10 ** (min_lr_log + (max_lr_log - min_lr_log) * lr) for lr in sobol_seq]

    # Evaluate initial guesses
    for lr in initial_guesses:
        loss = f(lr)
        evaluations.append((lr, loss))

    # Get next guess using GP
    while len(evaluations) < budget:
        # Get next guess using GP
        next_guess = get_next_guess(evaluations)

        # Evaluate next guess
        loss = f(next_guess)
        evaluations.append((next_guess, loss))

    best_lr = min(evaluations, key=lambda x: x[1])[0]
    return best_lr


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_loader, test_loader = load_fashion_mnist()

    # Optimize learning rate
    print("Optimizing learning rate...")
    best_lr = bayesian_lr_optimization(ResNet, train_model, train_loader, test_loader, epochs=10, budget=10)
    print(f"Best learning rate: {best_lr}")

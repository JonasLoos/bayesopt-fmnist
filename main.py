from typing import Callable
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import LogExpectedImprovement
import random
from matplotlib import pyplot as plt
import numpy as np


def fix_seed(seed: int = 42) -> None:
    '''
    Fixes the seed for reproducibility.

    Args:
        seed (int): The seed to use.
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fashion_mnist() -> tuple[Dataset, Dataset]:
    '''
    Loads the Fashion MNIST dataset.

    Returns:
        train_dataset (Dataset): The training set.
        test_dataset (Dataset): The test set.
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

    return train_dataset, test_dataset


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
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
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

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
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


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, epochs: int = 10,
                learning_rate: float = 0.001) -> float:
    '''
    Trains the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader):
            The dataloader for the train set.
        test_loader (DataLoader):
            The dataloader for the test set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
    '''
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data.to(device))
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"\rEpoch {epoch} | Batch {batch_idx} | "
                    f"Loss {loss.item():.4f}", end="", flush=True)

        print()

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for data, target in test_loader:
            output = model(data.to(device))
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            test_loss += criterion(output, target).item()

        test_loss /= len(test_loader)
        print(
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {100 * correct / total:.2f}%")

    return test_loss


def plot_bayesian_optimization(evaluations: list[tuple[float, float]],
                               min_lr_log: float, max_lr_log: float) -> None:
    '''
    Plots the Bayesian optimization process, including all observations,
    the posterior mean, the uncertainty estimate, and the acquisition function.

    Args:
        evaluations (list[tuple[float, float]]):
            The evaluations of the function.
        min_lr_log (float): The minimum learning rate in log10 scale.
        max_lr_log (float): The maximum learning rate in log10 scale.
    '''

    # Convert evaluations to numpy arrays
    X_obs_log = np.array([[np.log10(lr)] for lr, _ in evaluations])
    Y_obs = np.array([loss for _, loss in evaluations])

    # Scale X to unit cube [0, 1] for GP
    X_obs_scaled = ((X_obs_log - min_lr_log) / (max_lr_log - min_lr_log))

    # Create grid of points for visualization
    X_grid_log = np.linspace(min_lr_log, max_lr_log, 1000).reshape(-1, 1)
    X_grid_scaled = ((X_grid_log - min_lr_log) / (max_lr_log - min_lr_log))

    # Convert to tensors for GP
    X_tensor = torch.tensor(X_obs_scaled, dtype=torch.float64)
    Y_tensor = torch.tensor(Y_obs, dtype=torch.float64).reshape(-1, 1)

    # Store original scale parameters for later transformation
    y_mean = float(Y_tensor.mean())
    y_std = (float(Y_tensor.std()) if Y_tensor.std() > 0 else 1.0)

    # Standardize Y values
    Y_tensor_standardized = (Y_tensor - y_mean) / y_std

    # Fit GP model
    gp = SingleTaskGP(X_tensor, Y_tensor_standardized)

    # Get posterior
    X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float64)
    with torch.no_grad():
        posterior = gp.posterior(X_grid_tensor)
        mu_standardized = posterior.mean.numpy()
        std_standardized = posterior.variance.sqrt().numpy()

    # Transform predictions back to original scale
    mu = mu_standardized * y_std + y_mean
    std = std_standardized * y_std

    # Calculate acquisition function
    EI = LogExpectedImprovement(gp, best_f=Y_tensor_standardized.min())
    acquisition = (EI(X_grid_tensor.reshape(-1, 1, 1))
                   .detach().numpy())

    # Create plot
    fig, (ax_gp, ax_acq) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # GP posterior plot
    ax_gp.plot(X_grid_log, mu, 'k--', label='Prediction')
    ax_gp.fill_between(
        X_grid_log.ravel(),
        mu.ravel() - 1.96 * std.ravel(),
        mu.ravel() + 1.96 * std.ravel(),
        color='turquoise', alpha=0.4, label='95% CI'
    )
    ax_gp.scatter(X_obs_log, Y_obs, c='red', s=40, zorder=3,
                  label='Observations')
    ax_gp.set_ylabel('Loss')
    ax_gp.legend(loc='upper right')
    ax_gp.set_title('Gaussian Process and Acquisition Function')

    # Acquisition function plot
    ax_acq.plot(X_grid_log, acquisition, color='purple',
                label='Utility Function')
    ax_acq.set_xlabel('log10(Learning Rate)')
    ax_acq.set_ylabel('Utility')
    ax_acq.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def sobol_sequence(n: int) -> list[float]:
    '''
    Generates a 1-dimensional Sobol sequence.

    Args:
        n (int): The number of points to generate.

    Returns:
        sequence (list[float]):
            The Sobol sequence of length n in the interval [0, 1]
    '''

    sequence = []
    for i in range(n):
        x, f, k = 0.0, 0.5, i
        while k:
            x += f * (k & 1)   # add reversed bit
            k >>= 1            # shift right
            f *= 0.5           # next fraction
        sequence.append(x)
    return sequence


def get_next_guess(evaluations: list[tuple[float, float]],
                   min_lr_log: int, max_lr_log: int) -> float:
    '''
    Gets the next guess using the GP.

    Args:
        evaluations (list[tuple[float, float]]):
            The evaluations of the function.
        min_lr_log (int): The minimum learning rate in log10 scale.
        max_lr_log (int): The maximum learning rate in log10 scale.

    Returns:
        next_guess (float): The next guess.
    '''
    # Convert evaluations to tensors for botorch
    X_log = torch.tensor(
        [[torch.log10(torch.tensor(lr))] for lr, _ in evaluations],
        dtype=torch.float64)
    y = torch.tensor(
        [[loss] for _, loss in evaluations], dtype=torch.float64)

    # Scale X to unit cube [0, 1]
    X_scaled = (X_log - min_lr_log) / (max_lr_log - min_lr_log)

    # Standardize y values
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)

    # Initialize GP model (using default parameters)
    gp = SingleTaskGP(X_scaled, y)

    # Define acquisition function (Log Expected Improvement)
    best_f = y.min()
    EI = LogExpectedImprovement(gp, best_f=best_f)

    # Optimize acquisition function (bounds in unit cube)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    candidate, _ = optimize_acqf(
        EI, bounds=bounds,
        q=1, num_restarts=5, raw_samples=20,
    )

    # Convert back to learning rate scale
    candidate_log = candidate.item() * (max_lr_log - min_lr_log) + min_lr_log
    return 10 ** candidate_log


def bayesian_lr_optimization(
        model_class: type[nn.Module],
        train_function: Callable[[nn.Module, DataLoader,
                                 DataLoader, int, float],
                                 float],
        sobol_sequence: Callable[[int], list[float]],
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10, min_lr_log: int = -6, max_lr_log: int = -2,
        budget: int = 10, plot: bool = True) -> float:
    '''
    Bayesian Hyperparameter Optimization for the learning rate.

    Args:
        model_class (type[nn.Module]): The model class to optimize.
        train_function (Callable):
            The function to train the model.
        train_loader (DataLoader):
            The dataloader for the train set.
        test_loader (DataLoader):
            The dataloader for the test set.
        epochs (int): The number of epochs to train for.
        min_lr_log (int): The minimum learning rate in log10 scale.
        budget (int): The number of function evaluations.

    Returns:
        best_lr (float): The best learning rate.
    '''

    evaluations = []

    # Get initial guesses using Sobol sequence
    sobol_seq = sobol_sequence(min(3, budget))
    initial_guesses = [
        10 ** (min_lr_log + (max_lr_log - min_lr_log) * lr)
        for lr in sobol_seq
    ]

    # Evaluate initial guesses
    for lr in initial_guesses:
        loss = train_function(model_class(), train_loader, test_loader,
                              epochs, lr)
        evaluations.append((lr, loss))

    # Get next guess using GP
    while len(evaluations) < budget:

        if plot:
            plot_bayesian_optimization(evaluations, min_lr_log, max_lr_log)

        # Get next guess using GP
        next_guess = get_next_guess(evaluations, min_lr_log, max_lr_log)

        # Evaluate next guess
        loss = train_function(model_class(), train_loader, test_loader,
                              epochs, next_guess)
        evaluations.append((next_guess, loss))

    best_lr = min(evaluations, key=lambda x: x[1])[0]
    return best_lr


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_dataset_full, test_dataset = load_fashion_mnist()
    train_dataset, train_dataset_val = random_split(
        train_dataset_full, [0.8, 0.2])
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    train_loader_val = DataLoader(
        train_dataset_val, batch_size=128, shuffle=False)
    train_loader_full = DataLoader(
        train_dataset_full, batch_size=128, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False)

    # Optimize learning rate
    print("Optimizing learning rate...")
    best_lr = bayesian_lr_optimization(
        ResNet, train_model, train_loader, train_loader_val,
        epochs=10, budget=10)
    print(f"Best learning rate: {best_lr}")

    # Train model with best learning rate on full training set
    print("Training model with best learning rate on full training set...")
    model = ResNet()
    result_loss = train_model(model, train_loader_full, test_loader,
                              epochs=10, learning_rate=best_lr)

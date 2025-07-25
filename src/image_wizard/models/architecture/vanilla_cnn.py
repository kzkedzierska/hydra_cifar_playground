import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VanillaCNN"]


class VanillaCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.5,
        activation: str = "relu",
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.activation = getattr(F, activation)

        # Convolutional layers
        layers = []
        in_channels = 3  # RGB

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.features = nn.ModuleList(layers)

        # Calculate feature size after conv layers
        # CIFAR-10 is 32x32, after len(channels) pooling ops: 32 // (2^len(channels))
        feature_size = 32 // (2 ** len(channels))
        final_features = channels[-1] * feature_size * feature_size

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(final_features, 512), nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Apply conv layers with activation
        for i in range(0, len(self.features), 3):  # Conv, BN, Pool groups
            x = self.features[i](x)  # Conv
            x = self.features[i + 1](x)  # BatchNorm
            x = self.activation(x)  # Activation
            x = self.features[i + 2](x)  # MaxPool

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

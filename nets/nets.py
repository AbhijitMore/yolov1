import torch
import torch.nn as nn
from nets.nets_config import ARCHITECTURE_CONFIG, HYPERPARAMS

class CNNBlock(nn.Module):
    """
    A Convolutional Block that includes Conv2D, BatchNorm, and LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(HYPERPARAMS['leaky_relu_slope'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2D -> BatchNorm -> LeakyReLU."""
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    """
    YOLOv1 Model using a modular architecture.
    """
    def __init__(self, in_channels: int = HYPERPARAMS['in_channels'], 
                 split_size: int = HYPERPARAMS['split_size'], 
                 num_boxes: int = HYPERPARAMS['num_boxes'], 
                 num_classes: int = HYPERPARAMS['num_classes']):
        super(YoloV1, self).__init__()
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Create convolutional layers (Darknet)
        self.darknet = self._create_conv_layers(ARCHITECTURE_CONFIG)

        # Create fully connected layers for detection
        self.fcs = self._create_fcs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers and fully connected layers."""
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)

    def _create_conv_layers(self, architecture: list) -> nn.Sequential:
        """
        Constructs convolutional layers from the given architecture configuration.
        
        Args:
            architecture (list): A list defining the architecture of the model.

        Returns:
            nn.Sequential: Sequential block of convolutional layers.
        """
        layers = []
        in_channels = self.in_channels

        for item in architecture:
            if isinstance(item, tuple):
                # Add Convolutional Block
                layers.append(CNNBlock(in_channels, item[1], item[0], item[2], item[3]))
                in_channels = item[1]
            elif isinstance(item, str) and item == "M":
                # Add MaxPooling layer
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(item, list):
                # Add repeated blocks
                conv1, conv2, repeats = item
                for _ in range(repeats):
                    layers.append(CNNBlock(in_channels, conv1[1], conv1[0], conv1[2], conv1[3]))
                    layers.append(CNNBlock(conv1[1], conv2[1], conv2[0], conv2[2], conv2[3]))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self) -> nn.Sequential:
        """
        Constructs the fully connected layers for object detection.
        
        Returns:
            nn.Sequential: Fully connected layers.
        """
        S, B, C = self.split_size, self.num_boxes, self.num_classes
        output_dim = S * S * (C + B * 5)  # (S*S) * (C + B*5) -> Grid size * (Class + Bounding boxes)

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(HYPERPARAMS['dropout_rate']),
            nn.LeakyReLU(HYPERPARAMS['leaky_relu_slope']),
            nn.Linear(496, output_dim)
        )

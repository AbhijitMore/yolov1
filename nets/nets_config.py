# config.py

# Architecture configuration for YOLOv1
ARCHITECTURE_CONFIG = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 1),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# Hyperparameters
HYPERPARAMS = {
    'in_channels': 3,         # Number of input channels (3 for RGB images)
    'split_size': 7,          # Grid size (S)
    'num_boxes': 2,           # Number of bounding boxes (B)
    'num_classes': 20,        # Number of object classes (C)
    'dropout_rate': 0.0,      # Dropout rate
    'leaky_relu_slope': 0.1,  # LeakyReLU slope
}

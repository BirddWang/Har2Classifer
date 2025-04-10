import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models

class R3DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(R3DModel, self).__init__()
        # Load the pre-trained R3D model
        self.model = video_models.r3d_18(weights=None)
        # Modify the first convolutional layer to accept 5 input channels
        self.model.stem[0] = nn.Conv3d(in_channels=5,
                                       out_channels=self.model.stem[0].out_channels,
                                       kernel_size=self.model.stem[0].kernel_size,
                                       stride=self.model.stem[0].stride,
                                       padding=self.model.stem[0].padding,
                                       bias=self.model.stem[0].bias is not None)  # Keep bias setting consistent
        # Modify the final fully connected layer for the desired number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Example: 2 classes for Autism vs Control

    def forward(self, x):
        return self.model(x)
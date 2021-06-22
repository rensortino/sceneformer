import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        if "mnist" in w_path:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif "cifar" in w_path:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.load_state_dict(torch.load(w_path))
        self.linear = self.model.fc
        self.model.fc = nn.Identity()

    def forward(self, imgs):
        out_features = self.model(imgs)
        return out_features
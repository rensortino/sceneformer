import torch
import torch.nn as nn
import torchvision.transforms as T
# from torchvision import models
from resnet import resnet18

class ResNet18(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(num_classes=num_classes)
        if "mnist" in w_path:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif "cifar" in w_path:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.load_state_dict(torch.load(w_path))
        self.avg = self.resnet.avgpool
        self.linear = self.resnet.fc
        self.model = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))

        self.reduction = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, imgs):
        # out = self.resnet(imgs)
        out = self.model(imgs)
        # out = self.reduction(out)
        # out = out.flatten()
        # out = out.reshape(out.shape[0], -1)

        return out
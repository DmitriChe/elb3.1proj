from torch import nn
from torchvision.models import resnet152, ResNet152_Weights
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 1)
        for i in self.model.parameters():
            i.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
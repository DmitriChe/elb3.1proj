import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class MyResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        # подгружаем базовую модель ResNet50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # заменяем выходной слой - на выходе 6 классов
        self.model.fc = nn.Linear(2048, 6)
        # замораживаем все слои
        for i in self.model.parameters():
            i.requires_grad = False
        # размораживаем последний слой - для обучения
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

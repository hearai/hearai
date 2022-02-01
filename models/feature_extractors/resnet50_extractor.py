import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Resnet50Extractor(nn.Module):
    def __init__(self, representation_size=128):
        super().__init__()

        original_model = resnet50(pretrained=True, progress=True)
        in_features = 1000
        self.model = nn.Sequential(
            original_model,
            nn.Linear(in_features=in_features, out_features=representation_size),
        )

    def forward(self, x, **kwargs):
        return self.model(x)

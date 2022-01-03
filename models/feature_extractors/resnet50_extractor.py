import torch 
import torch.nn as nn
import timm

class Resnet50Extractor(nn.Module):
    def __init__(self, representation_size=256, model_path='resnet50'): 
        super().__init__() 
        self.model = timm.create_model(model_path, pretrained=True, num_classes=representation_size)
        
    def forward(self, x, **kwargs):
        return self.model(x)

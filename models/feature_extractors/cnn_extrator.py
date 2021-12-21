import torch 
import torch.nn as nn 
from torchvision import models 

original_model = models.resnet50(pretrained=True) 

class FeatureExtractorModule(nn.Module): 
    def __init__(self): 
        super(FeatureExtractorModule, self).__init__() 
        self.features = nn.Sequential(
             *list(original_model.features.children())[:-1] ) 
    def forward(self, x): 
        x = self.features(x) 
        return x 

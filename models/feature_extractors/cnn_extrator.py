#import timm
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

"""
in this version of scipt we:
- create tensor with random input
- run simple 2-layer linear network (which is not a target architecture, WiP)
- return a tensor of given size with random output, so it can be passed to the next step
"""

# TODO rewrite extractor definition to use a sequential and extract desired layer
# TODO when having target architecture, refactor all dependencies in other modules so whole pipeline can be run

class TwoLayerExctractor(torch.nn.Module):
    def __init__(self,input_dimension, hidden_dimension, output_dimension):
        """
        instantiate two nn.Linear modules
        """
        super(TwoLayerExctractor, self).__init__()
        self.linear1 = torch.nn.Linear(input_dimension, hidden_dimension)
        self.linear2 = torch.nn.Linear(hidden_dimension, output_dimension)

    def forward(self, x):
        """
        return tensor with prediction
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


batch_size, input_dimension, hidden_dimension, output_dimension = 128, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = Variable(torch.randn(batch_size, input_dimension))
y = Variable(torch.randn(batch_size, output_dimension))

model = TwoLayerExctractor(input_dimension, hidden_dimension, output_dimension)

y
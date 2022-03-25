import torch
import torch.nn as nn
from models.common.simple_sequential_model import SimpleSequentialModel

class Conv1DFeaturesProcessor(nn.Module):
    """ Basic sequential model for processing landmarks """

    def __init__(self,
                 representation_size: int = 512,
                 channels_factor: int = 2,
                 kernel_size: int = 5,
                 additional_layers: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.additional_layers = additional_layers
        self.representation_size = representation_size
        self.dropout_rate = dropout_rate

        self.representation_adjustment = nn.LazyLinear(out_features=representation_size)

        self.convolution = nn.Conv1d(in_channels=representation_size,
                                     out_channels=channels_factor * representation_size,
                                     groups=representation_size,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     padding_mode='replicate')

        self.simple_sequential = SimpleSequentialModel(layers=additional_layers,
                                                       representation_size=representation_size,
                                                       dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.representation_adjustment(x)

        x = torch.transpose(x, -2, -1)
        x = self.convolution(x)
        x = torch.transpose(x, -2, -1)

        x = self.simple_sequential(x)
        return x

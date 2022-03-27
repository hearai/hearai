import torch
import torch.nn as nn
from models.common.simple_sequential_model import SimpleSequentialModel

class Conv1DFeaturesProcessor(nn.Module):
    """ Basic sequential model for processing landmarks """

    def __init__(self,
                 representation_size: int = 512,
                 channels_factor: int = 3,
                 kernel_size: int = 3,
                 additional_layers: int = 3,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.additional_layers = additional_layers
        self.representation_size = representation_size
        self.dropout_rate = dropout_rate

        self.representation_adjustment = nn.LazyLinear(out_features=representation_size)

        self.convolutions = nn.ModuleList()
        out_multiplier = 1
        in_multiplier = 1
        for i in range(channels_factor):
            out_multiplier = (channels_factor - i) * out_multiplier
            self.convolutions.append(nn.Conv1d(in_channels=in_multiplier * representation_size,
                                     out_channels=out_multiplier * representation_size,
                                     groups=in_multiplier * representation_size,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     padding_mode='replicate'))
            self.convolutions.append(nn.ELU(0.1))
            self.convolutions.append(nn.Dropout(dropout_rate))
            in_multiplier = out_multiplier

        # self.convolutions.append(nn.Conv1d(in_channels=in_multiplier * representation_size,
        #                                    out_channels=representation_size,
        #                                    kernel_size=kernel_size,
        #                                    padding='same',
        #                                    padding_mode='replicate'))
        # self.convolutions.append(nn.ELU(0.1))
        # self.convolutions.append(nn.Dropout(dropout_rate))

        self.simple_sequential = SimpleSequentialModel(layers=additional_layers,
                                                       representation_size=representation_size,
                                                       dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.representation_adjustment(x)

        x = torch.transpose(x, -2, -1)

        for layer in self.convolutions:
            x = layer(x)

        x = torch.transpose(x, -2, -1)

        x = self.simple_sequential(x)
        return x

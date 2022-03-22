import torch.nn as nn
from models.common.simple_sequential_model import SimpleSequentialModel


class HeadClassificationSequentialModel(nn.Module):
    """ Basic sequential model for processing landmarks """

    def __init__(self,
                 classes_number: int,
                 representation_size: int = 512,
                 additional_layers: int = 0,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.classes_number = classes_number
        self.additional_layers = additional_layers
        self.representation_size = representation_size
        self.dropout_rate = dropout_rate
        self.layers_list = nn.ModuleList()
        if additional_layers > 0:
            self.layers_list.append(SimpleSequentialModel(layers=additional_layers,
                                                          representation_size=representation_size,
                                                          dropout_rate=dropout_rate))
        self.layers_list.append(nn.Linear(representation_size, classes_number))

    def forward(self, x, **kwargs):
        for layer in self.layers_list:
            x = layer(x)
        return x

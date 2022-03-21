import torch.nn as nn


class SimpleSequentialModel(nn.Module):
    """ Basic sequential model for processing landmarks """

    def __init__(self,
                 layers: int,
                 representation_size: int = 1024,
                 alpha: float = 0.1,
                 dropout_rate: float = 0.2):
        super().__init__()

        if layers < 1:
            raise Exception("Layers number should be an int greater or equal 1")

        self.layers = layers
        self.representation_size = representation_size
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        self.layers_list = nn.ModuleList(
            [
                nn.LazyLinear(representation_size),
                nn.ELU(alpha),
                nn.Dropout(dropout_rate)
            ]
        )
        for _ in range(1, layers):
            self.layers_list.append(nn.Linear(representation_size))
            self.layers_list.append(nn.ELU(alpha))
            self.layers_list.append(nn.Dropout(dropout_rate))

    def forward(self, x, **kwargs):
        for layer in self.layers_list:
            x = layer(x)
        return x

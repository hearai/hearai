import torch.nn as nn
from models.common.simple_sequencial_model import SimpleSequentialModel


class LandmarksSequentialModel(nn.Module):
    """ Basic sequential model for processing landmarks """

    def __init__(self, representation_size=1024, dropout_rate=0.2):
        super().__init__()

        self.representation_size = representation_size
        self.dropout_rate = dropout_rate
        self.model = SimpleSequentialModel(layers=3,
                                           representation_size=representation_size,
                                           dropout_rate=dropout_rate)

    def forward(self, x, **kwargs):
        return self.model(x)

import timm
import torch.nn as nn
import torch.optim


class VanillaTransformer(nn.Module):
    """Basic transformer model"""

    def __init__(self, input_size: int = 128, output_size: int = 1000):
        """
        Dummy example of __init__ function of basic transformer model. Does nothing as transformer is not even implemented.
        
        Args:
            input_size (int, optional): Input size fot the transformer model. Should be equal to the output size of the feature_extrator. Defaults to 128.
            output_size (int, optional): Output size. Should be equal to the selected representation size. Defaults to 1000.
        """
        super().__init__()
        self.__fully_connected = nn.Linear(input_size, output_size)
        self.__output_layer = nn.Softmax(dim=1)
        # this is a dummy example but in practice this will be longer

    def forward(self, input, **kwargs):
        x = self.__fully_connected(input)
        x = self.__output_layer(x)
        return x

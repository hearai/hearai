import timm
import torch.nn as nn


class VanillaTransformer(nn.Module):
    """Basic transformer model"""

    def __init__(self, input_size=128, output_size=1000):
        """
        Dummy example of __init__ function of basic transformer model. Does nothing as transformer is not even implemented.
        
        Args:
            input_size (int, optional): Input size fot the transformer model. Should be equal to the output size of the feature_extrator. Defaults to 128.
            output_size (int, optional): Output size. Should be equal to the selected representation size. Defaults to 1000.
        """
        super().__init__()
        self.transformer = nn.Linear(input_size, output_size)
        # this is a dummy example but in practice this will be longer 

    def forward(self, input, **kwargs):
        # this is a dummy example but in practice this will be longer 
        return self.transformer(input)

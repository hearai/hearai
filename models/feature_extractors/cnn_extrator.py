import timm
import torch.nn as nn

from torch.cuda import is_available as is_cuda_available


class CnnExtractor(nn.Module):
    """Basic timm model"""

    def __init__(self,
                 representation_size=128,
                 model_path="efficientnet_b1",
                 device='cpu'):
        """
        Dummy example of __init__ function of basic timm model. Simply loads a timm model and does nothing else.

        Args:
            representation_size (int, optional): Output size. Defaults to 128.
            model_path (str, optional): Path to timm pretrained model.
                    List of models is defined in timm docs. Defaults to 'efficientnet_b0'.
            device (str): Name of the device.
        """
        super().__init__()
        self.model = timm.create_model(
            model_path, pretrained=True, num_classes=representation_size
        )
        self.__device = device
        # this is a dummy example but in practice this will be longer

    def forward(self, input, **kwargs):
        # this is a dummy example but in practice this will be longer
        if self.__device == 'cpu':
            return self.model(input.cpu())
        else:
            return self.model(input.cuda())


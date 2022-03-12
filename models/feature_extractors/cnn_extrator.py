import timm
import torch.nn as nn


class CnnExtractor(nn.Module):
    """Basic timm model"""

    def __init__(
        self, representation_size=128, model_path="efficientnet_b1"
    ):
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
        # this is a dummy example but in practice this will be longer

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = (i >= 297)

    def forward(self, input, **kwargs):
        # this is a dummy example but in practice this will be longer
        return self.model(input)

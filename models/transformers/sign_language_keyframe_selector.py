import math
from typing import Dict

import torch
import torch.nn as nn

from models.common.simple_sequential_model import SimpleSequentialModel


class SignLanguageKeyframeSelector(nn.Module):
    """
    Sign Language Transformer
    https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf
    """

    __max_len = 10000

    def __init__(
        self,
        feature_extractor_parameters: Dict = None,
        transformer_parameters: Dict = None,
        train_parameters: Dict = None
    ):
        """
        Args:
            feature_extractor_parameters (Dict): Dict containing parameters regarding currently used feature extractor.
                [Warning] Must contain fields: 
                    - "representation_size" (int)
            transformer_parameters (Dict): Dict containing parameters regarding currently used transformer.
                [Warning] Must containt fields:
                    - "output_size" (int)
                    - "feedforward_size" (int)
                    - "num_encoder_layers" (int)
                    - "num_attention_heads" (int)
                    - "dropout_rate" (float)
            train_parameters (Dict): Dict containing parameters of the training process.
                [Warning] Must containt fields:
                    - "num_segments" (int)
        """
        super(SignLanguageKeyframeSelector, self).__init__()
        self._input_size = feature_extractor_parameters["representation_size"]
        self._attention_heads = transformer_parameters["num_attention_heads"]


        self._cnn_layers = nn.Conv2d(in_channels=self._attention_heads,
                                     out_channels=self._attention_heads,
                                     groups=self._attention_heads,
                                     kernel_size=(5, 1),
                                     padding='same',
                                     padding_mode='replicate')

        self._importance_calculator = nn.Conv2d(in_channels=self._attention_heads,
                                     out_channels=self._attention_heads,
                                     groups=self._attention_heads,
                                     kernel_size=(1, 2 * self._input_size),
                                     padding='valid',
                                     padding_mode='replicate')
        self._importance_dropout = nn.Dropout(transformer_parameters["dropout_rate"])
        self._importance_activation = nn.Softmax(dim=-2)

        self._sequential = SimpleSequentialModel(layers=2,
                                                 representation_size=transformer_parameters["output_size"],
                                                 dropout_rate=transformer_parameters["dropout_rate"])

    def forward(self, input: torch.Tensor):

        shape = input.shape
        input = torch.reshape(input, (shape[0], 1, shape[1], shape[2]))

        if self._attention_heads > 1:
            input = torch.concat([input for _ in range(self._attention_heads)], 1)

        x = self._cnn_layers(input)

        x = torch.concat(
            [
                input,
                torch.multiply(x, x)
            ],
            dim=-1
        )
        x = self._importance_dropout(x)
        x = self._importance_calculator(x)
        x = self._importance_activation(x)

        x = torch.multiply(x, input)
        x = torch.sum(x, dim=-2)

        x = torch.flatten(x, start_dim=1)

        x = self._sequential(x)

        return x

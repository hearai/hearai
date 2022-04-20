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
        self._num_encoder_layers = transformer_parameters["num_encoder_layers"]
        self._dropout_rate = transformer_parameters["dropout_rate"]
        self._mix_coordinates = False

        self._velocity_modules = nn.ModuleList()
        self._mixing_layers = nn.ModuleList()
        self._acceleration_modules = nn.ModuleList()

        for i in range(self._num_encoder_layers):
            velocity_module = nn.ModuleList()
            velocity_module.append(
                nn.Conv2d(in_channels=self._attention_heads,
                          out_channels=self._attention_heads,
                          groups=self._attention_heads,
                          kernel_size=(3, 8),
                          stride=(1, 8),
                          padding=(1, 0),
                          padding_mode='replicate')
            )
            velocity_module.append(
                nn.ELU(0.1)
            )
            self._velocity_modules.append(velocity_module)

            if i > 0:
                acceleration_module = nn.ModuleList()
                acceleration_module.append(
                    nn.Conv2d(in_channels=self._attention_heads,
                              out_channels=self._attention_heads,
                              groups=self._attention_heads,
                              kernel_size=(3, 1),
                              padding='same',
                              padding_mode='replicate')
                )
                acceleration_module.append(
                    nn.ELU(0.1)
                )
                self._acceleration_modules.append(acceleration_module)

            if self._mix_coordinates:
                mixing_module = nn.ModuleList()
                mixing_module.append(
                    nn.Dropout(self._dropout_rate)
                )
                mixing_module.append(
                    nn.Conv2d(in_channels=self._attention_heads,
                              out_channels=self._attention_heads,
                              groups=self._attention_heads,
                              kernel_size=(1, int((i + 1) * self._input_size / 8)),
                              padding='same',
                              padding_mode='replicate')
                )
                mixing_module.append(
                    nn.ELU(0.1)
                )
                self._mixing_layers.append(mixing_module)

        self._importance_dropout = nn.Dropout(self._dropout_rate)
        self._importance_calculator = nn.Conv2d(
            in_channels=self._attention_heads,
            out_channels=self._attention_heads,
            groups=self._attention_heads,
            kernel_size=(1, int(self._input_size * (0 + self._num_encoder_layers / 8)))
        )
        self._importance_activation = nn.Softmax(dim=-2)

        # self._final_sequential = SimpleSequentialModel(layers=2,
        #                                                representation_size=transformer_parameters["output_size"],
        #                                                dropout_rate=transformer_parameters["dropout_rate"])

    def forward(self, input: torch.Tensor):

        shape = input.shape
        input0 = torch.reshape(input, (shape[0], 1, shape[1], shape[2]))

        if self._attention_heads > 1:
            input0 = torch.concat([input0 for _ in range(self._attention_heads)], 1)

        for i in range(self._num_encoder_layers):
            v = input0
            for v_layer in self._velocity_modules[i]:
                v = v_layer(v)

            if i == 0:
                a = v
            else:
                for a_layer in self._acceleration_modules[i - 1]:
                    a = a_layer(a)
                a = torch.concat([v, a], dim=-1)

            if self._mix_coordinates:
                for m_layer in self._mixing_layers[i]:
                    a = m_layer(a)

        w = self._importance_dropout(a)
        w = self._importance_calculator(w)
        w = self._importance_activation(w)

        x = torch.multiply(w, input0)
        x = torch.sum(x, dim=-2)
        x = torch.flatten(x, start_dim=1)

        # x = self._final_sequential(x)

        return x

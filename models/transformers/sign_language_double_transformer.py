import math
from typing import Dict

import torch
import torch.nn as nn

from models.feature_extractors.conv1d_features_processor import Conv1DFeaturesProcessor


class SignLanguageDoubleTransformer(nn.Module):
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
        super(SignLanguageDoubleTransformer, self).__init__()
        self._input_size = feature_extractor_parameters["representation_size"]

        self._transformers_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=self._input_size,
                                           nhead=transformer_parameters["num_attention_heads"],
                                           dim_feedforward=transformer_parameters["feedforward_size"],
                                           dropout=transformer_parameters["dropout_rate"],
                                           activation='gelu',
                                           batch_first=True)
                for _ in range(transformer_parameters["num_encoder_layers"])
            ]
        )
        self._reversed_transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=train_parameters["num_segments"],
                                           nhead=transformer_parameters["num_attention_heads"],
                                           dim_feedforward=train_parameters["num_segments"],
                                           dropout=transformer_parameters["dropout_rate"],
                                           activation='gelu',
                                           batch_first=True)
                for _ in range(transformer_parameters["num_encoder_layers"])
            ]
        )
        self._dropout_positional_encoding = nn.Dropout(transformer_parameters["dropout_rate"])
        self._last_norm = nn.LayerNorm(self._input_size)
        self._last_linear = nn.Linear(train_parameters["num_segments"] * self._input_size, transformer_parameters["output_size"])

        self._position_encoding = self.__get_position_encoding()

    def forward(self, input: torch.Tensor):
        # Positional Encoding Start
        positional_encoding = input + self._position_encoding[:, : input.shape[1]].to(input.device)

        x = self._dropout_positional_encoding(positional_encoding)
        # Positional Encoding End

        for transformer_layer, reversed_transformer_layer in zip(self._transformers_layers, self._reversed_transformer_layers):
            x = torch.transpose(x, -1, -2)
            x = reversed_transformer_layer(x)
            x = torch.transpose(x, -1, -2)
            x = transformer_layer(x)

        x = self._last_norm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self._last_linear(x)
        return x

    def __get_position_encoding(self):
        position_encoding = torch.zeros(self.__max_len, self._input_size)
        position = torch.arange(0, self.__max_len).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, self._input_size, 2, dtype=torch.float)
                * -(math.log(10000.0) / self._input_size)
            )
        )
        position_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        position_encoding[:, 1::2] = torch.cos(position.float() * div_term)
        return position_encoding.unsqueeze(0)
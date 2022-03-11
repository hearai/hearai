from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn

from transformers import HubertModel, HubertConfig


class HubertTransformer(nn.Module):
    """
    Pretrained transformer model from huggingface repository.
    """

    def __init__(
        self,
        feature_extractor_parameters: Dict = None,
        transformer_parameters: Dict = None,
        train_parameters: Dict = None,
        # input_features: int = 2048,
        # output_features: int = 512,
        # num_segments: int = 10,
        *args,
        **kwargs
    ):
        """
        TODO: update docstring
        Args:
            input_features (int): Argument to keep name convention consistency.
            output_features (int): Expected number of output features from the model.
            num_segments (int): Number of frames associated with single sign.
        """
        super(HubertTransformer, self).__init__()
        configuration = HubertConfig()
        self.model = HubertModel(configuration,)
        self._input_features = feature_extractor_parameters["representation_size"]
        self._output_features = transformer_parameters["output_size"]

        # Define layers
        hidden_features = self.find_hidden_features_number(
            input_size=self._input_features * train_parameters["num_segments"],
            hidden_size=configuration.hidden_size,
            kernels=configuration.conv_kernel,
            strides=configuration.conv_stride,
        )
        self._linear = nn.Linear(
            in_features=hidden_features, out_features=self._output_features
        )

    def forward(self, input: torch.Tensor, **kwargs):
        if len(input.shape) > 2:
            x = torch.reshape(input, (input.shape[0], -1))
        else:
            x = input
        x = self.model(x).last_hidden_state
        x = torch.reshape(x, (x.shape[0], -1))
        x = self._linear(x)

        return x

    def find_hidden_features_number(
        self, input_size: int, hidden_size: int, kernels: List, strides: List
    ):
        def compute_size(input, kernel, stride):
            return ((input - kernel) / stride) + 1

        def compute_size_list(input, kernels, strides):
            new_input = input
            for kernel, stride in zip(kernels, strides):
                new_input = compute_size(new_input, kernel, stride)
            return new_input

        hidden_features_number = (
            np.floor(
                compute_size_list(input=input_size, kernels=kernels, strides=strides)
            )
            * hidden_size
        ).astype(int)
        return hidden_features_number

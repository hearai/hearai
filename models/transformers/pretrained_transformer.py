from typing import List

import torch
import torch.nn as nn

from transformers import HubertModel


class HubertTransformer(nn.Module):
    """
    Pretrained transformer model from huggingface repository.
    """

    def __init__(self,
                 input_features: int = 2048,
                 output_features: int = 512,
                 *args,
                 **kwargs):
        """
        Args:
            input_features (int): Argument to keep name convention consistency.
            output_features (int): Expected number of output features from the model.
        """
        super(HubertTransformer, self).__init__()
        checkpoint = 'facebook/hubert-large-ls960-ft'
        self.model = HubertModel.from_pretrained(checkpoint)
        self.model.train()
        self.__output_features = output_features

    def forward(self, input: torch.Tensor, **kwargs):
        if len(input.shape) > 2:
            x = torch.reshape(input, (input.shape[0], -1))
        else:
            x = input
        # model = self.__model
        x = self.model(x).last_hidden_state
        x = torch.reshape(x, (x.shape[0], -1))
        x = nn.Linear(in_features=x.shape[1], out_features=self.__output_features)(x)

        return x

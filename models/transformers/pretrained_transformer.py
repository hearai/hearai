from typing import List

import torch
import torch.nn as nn

from transformers import HubertModel


class PretrainedTransformer(nn.Module):
    """
    Pretrained transformer model from huggingface repository.
    """

    def __init__(self,
                 checkpoint='facebook/hubert-large-ls960-ft',
                 **kwargs):
        """
        Args:
            checkpoint (str): model's checkpoint name in PyTorch transformers repository.
        """
        super(PretrainedTransformer, self).__init__()

        self.__model = HubertModel.from_pretrained(checkpoint)

    def forward(self, input: torch.Tensor, **kwargs):
        if len(input.shape) > 2:
            x = torch.reshape(input, (input.shape[0], -1))
        else:
            x = input
        model = self.__model
        x = model(x).last_hidden_state
        x = torch.reshape(x, (x.shape[0], -1))

        return x

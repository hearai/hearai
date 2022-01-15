from typing import List

import torch
import torch.nn as nn

from transformers import HubertModel


class PretrainedTransformer(nn.Module):
    """
    Pretrained transformer model from huggingface repository.
    """

    def __init__(self,
                 checkpoint='',
                 encoder_list: List[int] = [],
                 add_softmax: bool = False):
        """
        Args:
            checkpoint (str): model's checkpoint name in PyTorch transformers repository.
            encoder_list (List[int]): List of integers indicating how many final Linear lyers will be appended at the
                    end of the model.
            add_softmax (bool): if True Softmax layer will be added at the end so enable classification mode.
        """
        super(PretrainedTransformer, self).__init__()

        self.__encoder_list = encoder_list
        self.__add_softmax = add_softmax
        self.__model = HubertModel.from_pretrained(checkpoint)

    def forward(self, input: torch.Tensor, **kwargs):
        if len(input.shape) > 2:
            x = torch.reshape(input, (input.shape[0], -1))
        else:
            x = input
        model = self.__model
        x = model(x).last_hidden_state
        x = torch.reshape(x, (x.shape[0], -1))
        for encoder_size in self.__encoder_list:
            x = nn.Linear(in_features=x.shape[-1], out_features=encoder_size)(x)

        if self.__add_softmax:
            x = nn.Softmax()(x)

        return x

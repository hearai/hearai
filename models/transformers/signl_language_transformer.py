import math

import torch
import torch.nn as nn


class SignLanguageTransformer(nn.Module):
    """
    Sign Language Transformer
    https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self):
        super(SignLanguageTransformer, self).__init__()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input (torch.Tensor): input of shape (batch_size, number_of_embeddings, number_of_features)

        Returns:

        """
        # Positional encoding
        size = 512  # num_features
        max_len = 10000
        position_encoding = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        position_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        position_encoding[:, 1::2] = torch.cos(position.float() * div_term)
        position_encoding = position_encoding.unsqueeze(0)

        positional_encoding = input + position_encoding[:, : input.shape[1]]

        # todo: Self Attention (Multi-Head Attention)
        positional_encoding_normalized = nn.LayerNorm(positional_encoding)

        values = positional_encoding_normalized
        keys = positional_encoding_normalized
        questions = positional_encoding_normalized

        attention_output = MultiHeadAttention()(v=values,
                                                k=keys,
                                                q=questions)

        # todo: Add & Norm

        # todo: Feedforward

        # todo: Add & Norm

        #

        pass


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def forward(self,
                v: torch.Tensor,
                k: torch.Tensor,
                q: torch.Tensor):
        """
        Implementation follows paper: https://arxiv.org/pdf/1706.03762.pdf

        Args:
            v (torch.Tensor): Values tensor.
            k (torch.Tensor): Keys tensor.
            q (torch.Tensor): Queries tensor.

        Returns:
            (torch.Tensor)
        """
        # todo: @Alicja
        pass


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self,
                v: torch.Tensor,
                k: torch.Tensor,
                q: torch.Tensor):
        """
        Implementation follows paper: https://arxiv.org/pdf/1706.03762.pdf

        Args:
            v (torch.Tensor): Values tensor.
            k (torch.Tensor): Keys tensor.
            q (torch.Tensor): Queries tensor.

        Returns:
            (torch.Tensor)
        """
        # todo: @Alicja
        pass


if __name__ == '__main__':
    import torch
    import numpy as np

    tmp = np.random.random((1, 4, 512))
    input = torch.Tensor(tmp)

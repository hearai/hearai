import math

import torch
import torch.nn as nn


class SignLanguageTransformer(nn.Module):
    """
    Sign Language Transformer
    https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf
    """

    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 1024,
                 feedforward_size: int = 1024,
                 num_encoder_layers: int = 1,
                 num_frames: int = 10,
                 dropout_rate: float = 0.1,):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            feedforward_size (int): Number of features in intermediate feedforward "layer".
            num_encoder_layers (int): Number of encoder layers.
            num_frames (int): Number of frames in a video.
            dropout_rate (float): Dropout rate.
        """
        super(SignLanguageTransformer, self).__init__()
        self._dropout_rate = dropout_rate
        self._feedforward_size = feedforward_size
        self._input_size = input_size
        self._num_encoder_layers = num_encoder_layers

        self._slrt_layers = nn.ModuleList([SLRTEncoder(input_size=input_size,
                                                       feedforward_size=feedforward_size,
                                                       num_frames=num_frames,
                                                       dropout_rate=dropout_rate)])
        self._dropout_positional_encoding = nn.Dropout(dropout_rate)
        self._last_norm = nn.LayerNorm(input_size)
        self._last_linear = nn.Linear(num_frames*input_size, output_size)

    def forward(self, input: torch.Tensor):
        # Positional Encoding Start
        max_len = 10000
        position_encoding = torch.zeros(max_len, self._input_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, self._input_size, 2, dtype=torch.float) * -(math.log(10000.0) / self._input_size))
        )
        position_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        position_encoding[:, 1::2] = torch.cos(position.float() * div_term)
        position_encoding = position_encoding.unsqueeze(0)

        positional_encoding = input + position_encoding[:, :input.shape[1]]
        x = self._dropout_positional_encoding(positional_encoding)
        # Positional Encoding End

        for slrt_layer in self._slrt_layers:
            x = slrt_layer(x)

        x = self._last_norm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self._last_linear(x)
        return x


class SLRTEncoder(nn.Module):
    """
    Single Encoder layer for Sign Language Transformer.
    """
    def __init__(self,
                 input_size: int,
                 feedforward_size: int,
                 num_frames: int,
                 dropout_rate: float):
        """
        Args:
            input_size (int): Number of input features.
            feedforward_size (int): Number of features in intermediate feedforward "layer".
            num_frames (int): Number of frames per video.
            dropout_rate (float): Dropout rate.
        """
        super(SLRTEncoder, self).__init__()

        self._input_size = input_size
        self._num_frames = num_frames
        self._dropout_rate = dropout_rate

        self._positional_encoding_norm = nn.LayerNorm(input_size)
        self._attention_dropout = nn.Dropout(dropout_rate)
        self._feedforward_sequential = nn.Sequential(nn.LayerNorm(input_size),
                                                     nn.Linear(input_size, feedforward_size),
                                                     nn.ReLU(),
                                                     nn.Dropout(dropout_rate),
                                                     nn.Linear(feedforward_size, input_size),
                                                     nn.Dropout(dropout_rate))

    def forward(self, input: torch.Tensor):
        # SLRT Start
        positional_encoding_normalized = self._positional_encoding_norm(input)

        # Self Attention (Multi-Head Attention) Start
        values = positional_encoding_normalized
        keys = positional_encoding_normalized
        questions = positional_encoding_normalized

        attention_output = MultiHeadedAttention(num_heads=self._num_frames,
                                                size=self._input_size,
                                                dropout_rate=self._dropout_rate)(v=values,
                                                                                 k=keys,
                                                                                 q=questions)

        attention_output = self._attention_dropout(attention_output)
        # Self Attention (Multi-Head Attention) End

        # Feedforward Start
        feedforward_x = self._feedforward_sequential(attention_output)
        x = feedforward_x + attention_output
        # Feedforward End

        # SLRT End
        return x


class MultiHeadedAttention(nn.Module):
    """
    Implementation based on
    https://github.com/neccam/slt
    """

    def __init__(self,
                 num_heads: int,
                 size: int,
                 dropout_rate: float = 0.1):
        """
        Layer implemented as presented in the paper "Attention is all you need"
        https://arxiv.org/pdf/1706.03762.pdf
        Args:
            num_heads (int): Number of input images per single video.
            size (int): Number of features per head.
            dropout_rate (float): Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self,
                k: torch.Tensor,
                v: torch.Tensor,
                q: torch.Tensor,
                mask: torch.Tensor = None):
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self._dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


if __name__ == '__main__':
    import torch
    import numpy as np

    tmp = np.random.random((1, 4, 512))
    input = torch.Tensor(tmp)
    input_size = 512
    dropout_rate = 0.1
    feedforward_size = 128

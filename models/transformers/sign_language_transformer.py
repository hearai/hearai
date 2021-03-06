import math
from typing import Dict

import torch
import torch.nn as nn


class SignLanguageTransformer(nn.Module):
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
        super(SignLanguageTransformer, self).__init__()
        self._input_size = feature_extractor_parameters["representation_size"]

        self._slrt_layers = nn.ModuleList(
            [
                SLRTEncoder(
                    input_size=self._input_size,
                    feedforward_size=transformer_parameters["feedforward_size"],
                    num_attention_heads=transformer_parameters["num_attention_heads"],
                    dropout_rate=transformer_parameters["dropout_rate"],
                )
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

        for slrt_layer in self._slrt_layers:
            x = slrt_layer(x)

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


class SLRTEncoder(nn.Module):
    """
    Single Encoder layer for Sign Language Transformer.
    """

    def __init__(
        self,
        input_size: int,
        feedforward_size: int,
        num_attention_heads: int,
        dropout_rate: float,
    ):
        """
        Args:
            input_size (int): Number of input features.
            feedforward_size (int): Number of features in intermediate feedforward "layer".
            num_attention_heads (int): Number of heads attention layer.
            dropout_rate (float): Dropout rate.
        """
        super(SLRTEncoder, self).__init__()

        self._positional_encoding_norm = nn.LayerNorm(input_size)
        self._attention_dropout = nn.Dropout(dropout_rate)

        self._multi_headed_attention = MultiHeadedAttention(
            num_heads=num_attention_heads,
            size=input_size,
            dropout_rate=dropout_rate,
        )
        self._feedforward_sequential = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, feedforward_size),
            nn.ELU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_size, input_size),
            nn.ELU(0.1),
            nn.Dropout(dropout_rate),
        )



    def forward(self, input: torch.Tensor):

        positional_encoding_normalized = self._positional_encoding_norm(input)
        # Self Attention (Multi-Head Attention) Start
        values = positional_encoding_normalized
        keys = positional_encoding_normalized
        questions = positional_encoding_normalized

        attention_output = self._multi_headed_attention(v=values, k=keys, q=questions)
        attention_output = self._attention_dropout(attention_output)

        # Self Attention (Multi-Head Attention) End
        x = input + attention_output
        feedforward_x = self._feedforward_sequential(x)
        # Feedforward End

        # SLRT End
        return feedforward_x


class MultiHeadedAttention(nn.Module):
    """
    Implementation based on
    https://github.com/neccam/slt
    """

    def __init__(self, num_heads: int, size: int, dropout_rate: float = 0.1):
        """
        Layer implemented as presented in the paper "Attention is all you need"
        https://arxiv.org/pdf/1706.03762.pdf
        Args:
            num_heads (int): Number of output heads.
            size (int): Number of features per head.
            dropout_rate (float): Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()

        self.head_size = head_size = size
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, self.num_heads * head_size)
        self.v_layer = nn.Linear(size, self.num_heads * head_size)
        self.q_layer = nn.Linear(size, self.num_heads * head_size)

        self.output_layer = nn.Linear(self.num_heads * size, size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size = k.size(0)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)

        # get context vector (select values with attention) and reshape
        # back to [number of items in a batch, number of segments * number of frames per segment, number of features]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output

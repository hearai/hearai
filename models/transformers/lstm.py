from typing import List, Dict
import torch


class LSTM(torch.nn.Module):
    """Basic LSTM model"""

    def __init__(
        self,
        feature_extractor_parameters: Dict = None,
        transformer_parameters: Dict = None,
        train_parameters: Dict = None,
        *args,
        **kwargs
    ):
        """
        Lightweight basic LSTM model.
        
        Required args (should be passed in a Dict):
            feature_extractor_parameters["representation_size"] as input_size (int): Input size fot the LSTM model. Should be equal to the output size of the feature_extrator.
            transformer_parameters["output_size"] as output_size (int): Output size. Should be equal to the selected representation size. 
            train_parameters["num_segments"] as num_frames (int): Number of input frames.
            transformer_parameters["feedforward_size"] as feedforward_size (int): Hidden size of an LSTM model. Suggested to 512.
            transformer_parameters["num_encoder_layers"] as num_encoder_layers (int): Number of LSTM layers. Suggested to 2.
            transformer_parameters["dropout_rate"] as dropout_rate (float): Dropout stats. Should be between 0 and 1. Suggested 0.2.
        """
        super().__init__()
        self.input_size = feature_extractor_parameters["representation_size"]
        self.num_frames = train_parameters["num_segments"]
        self.output_size = transformer_parameters["output_size"]
        self.feedforward_size = transformer_parameters["feedforward_size"]
        self.num_encoder_layers = transformer_parameters["num_encoder_layers"]
        self.dropout_rate = transformer_parameters["dropout_rate"]

        self.LSTM = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.feedforward_size,
            num_layers=self.num_encoder_layers,
            dropout=self.dropout_rate,
            bidirectional=False,
        )
        self.hidden_output_size = self.feedforward_size * self.num_frames
        self.fully_connected = torch.nn.Linear(
            self.hidden_output_size, self.output_size
        )

    def forward(self, input, **kwargs):
        x = self.LSTM(input.reshape(self.num_frames, -1, self.input_size))[0]
        out = self.fully_connected(x.reshape(-1, self.hidden_output_size))
        return out

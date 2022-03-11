import torch
class LSTM(torch.nn.Module):
    """Basic transformer model"""

    def __init__(self, input_size: int,
                 output_size: int,
                 feedforward_size,
                 num_encoder_layers,
                 num_frames,
                 dropout_rate):
        """
        Dummy example of __init__ function of basic transformer model. Does nothing as transformer is not even implemented.
        
        Args:
            input_size (int, optional): Input size fot the transformer model. Should be equal to the output size of the feature_extrator. Defaults to 128.
            output_size (int, optional): Output size. Should be equal to the selected representation size. Defaults to 1000.
        """
        super().__init__()
        self.LSTM = torch.nn.LSTM(input_size=input_size,
                            hidden_size=feedforward_size,
                            num_layers=num_encoder_layers,
                            dropout=dropout_rate,
                            bidirectional=False)
        self.hidden_output_size = feedforward_size*num_frames
        self.input_size = input_size
        self.num_frames = num_frames
        self.fully_connected = torch.nn.Linear(self.hidden_output_size, output_size)
        

    def forward(self, input, **kwargs):
        x = self.LSTM(input.reshape(self.num_frames, -1, self.input_size))[0]
        out = self.fully_connected(x.reshape(-1,self.hidden_output_size))
        return out
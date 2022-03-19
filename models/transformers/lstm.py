import torch
class LSTM(torch.nn.Module):
    """Basic LSTM model"""

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 num_frames: int,
                 feedforward_size = 512,
                 num_encoder_layers = 2,
                 dropout_rate = 0.2):
        """
        Lightweight function of basic LSTM model. Not a transformer but nice to test.
        
        Args:
            input_size (int): Input size fot the LSTM model. Should be equal to the output size of the feature_extrator.
            output_size (int): Output size. Should be equal to the selected representation size. 
            num_frames (int): Number of input frames.
            feedforward_size (int): Hidden size of an LSTM model. Defaults to 512.
            num_encoder_layers (int): Number of LSTM layers. Defaults to 2.
            dropout_rate (float): Dropout stats. Should be between 0 and 1. Deafults to 0.2.
            
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

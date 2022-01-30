import torch
import torch.nn as nn

from torch.autograd import Variable


class MultiFrameFeatureExtractor(nn.Module):
    """
    Class to extract features from every frame of the video.
    Input to the model should have shape BxNxCxHxW (batch x frames x channels x height x width).
    Output from the model is a Tensor of shape NxM (batch x representation_size).

    Args:
        feature_extractor_name (str): the feature extractor.
    """

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
    def forward(self, x):
        frame_features = []

        for frame in x:
            output = self.feature_extractor(frame).squeeze(0)
            frame_features.append(Variable(output))

        frame_features = torch.stack(frame_features, dim=0)
        return frame_features

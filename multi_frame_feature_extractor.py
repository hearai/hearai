import torch
import torch.nn as nn
from models.model_loader import ModelLoader

class MultiFrameFeatureExtractor(nn.Module):
    """
    Class to extract features from every frame of the video.

    Input to the model should have shape BxNxCxHxW (batch x frames x channels x height x width).

    Output from the model is a list contains Tensors of shape NxM (count_of_frames x representation_size).
    Every Tensor is a one batch.

    Args:
        feature_extractor_name (str): a name of the feature extractor.
        representation_size (int): a size of input to transformer.
        model_path (str): a name of the model in timm.
    """

    def __init__(self,
                 feature_extractor_name,
                 representation_size,
                 model_path):

        super().__init__()

        
        self.feature_extractor = ModelLoader().load_feature_extractor(feature_extractor_name, representation_size, model_path)


    def forward(self, x):
        outputs = []

        for batch in x:
            frame_features = []

            for frame in batch:
                frame = frame.unsqueeze(0)
                output = self.feature_extractor(frame).squeeze(0)
                frame_features.append(output)

            frame_features = torch.stack(frame_features, dim=0)
            outputs.append(frame_features)

        return outputs

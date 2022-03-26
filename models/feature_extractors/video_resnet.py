from typing import Any

from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D


def CustomVideoResNet(num_classes, **kwargs: Any):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248
    Example configuartions are in
    https://pytorch.org/vision/stable/_modules/torchvision/models/video/resnet.html
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        num_classes=num_classes,
        **kwargs,
    )

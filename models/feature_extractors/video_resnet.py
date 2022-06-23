from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, Conv3DNoTemporal, BasicStem, R2Plus1dStem, Conv2Plus1D
from torch.hub import load_state_dict_from_url

model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


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

def _video_resnet(arch: str, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VideoResNet:
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def r3d_18(num_classes, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet(
        "r3d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        num_classes=num_classes,
        **kwargs,
    )

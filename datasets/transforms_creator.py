import torchvision.transforms as T

from datasets.dataset import ImglistToTensor


class TransformsCreator:
    """
    Class for transforms creation and parametrization.
    """

    def __init__(self,
                 apply_resize: bool = True,
                 apply_center_crop: bool = True,
                 apply_random_erasing: bool = True,
                 apply_random_rotation: bool = True,
                 apply_color_jitter: bool = True,

                 resize_size: int = 256,
                 center_crop_size: int = 256,
                 random_erasing_probability: float = 0.5,
                 random_rotation_degree: int = 5,
                 color_jitter_brightness: float = 0.1,
                 color_jitter_contrast: float = 0.1,
                 color_jitter_saturation: float = 0.1,
                 color_jitter_hue: float = 0.05):

        self.apply_resize = apply_resize
        self.apply_center_crop = apply_center_crop
        self.apply_random_erasing = apply_random_erasing
        self.apply_random_rotation = apply_random_rotation
        self.apply_color_jitter = apply_color_jitter

        self.resize_size = resize_size

        self.center_crop_size = center_crop_size

        self.random_erasing_probability = random_erasing_probability

        self.random_rotation_degree = random_rotation_degree

        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

    def get_train_transforms(self) -> T.Compose:
        pil_augmentations = []
        tensor_augmentations = []

        if self.apply_random_erasing:
            tensor_augmentations.append(T.RandomErasing(self.random_erasing_probability))

        if self.apply_random_rotation:
            tensor_augmentations.append(T.RandomRotation(degrees=self.random_rotation_degree))

        if self.apply_color_jitter:
            tensor_augmentations.append(T.ColorJitter(brightness=self.color_jitter_brightness,
                                                      contrast=self.color_jitter_contrast,
                                                      saturation=self.color_jitter_saturation,
                                                      hue=self.color_jitter_hue))

        return self._get_transforms(pil_augmentations, tensor_augmentations)

    def get_val_transforms(self) -> T.Compose:
        pil_augmentations = []
        tensor_augmentations = []
        return self._get_transforms(pil_augmentations, tensor_augmentations)

    def _get_transforms(self, pil_augmentations: list, tensor_augmentations: list) -> T.Compose:
        return T.Compose(
            [
                *pil_augmentations,
                ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
                *tensor_augmentations,
                T.Resize(self.resize_size),  # image batch, resize smaller edge to 256
                T.CenterCrop(self.center_crop_size),  # image batch, center crop to square 256x256
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

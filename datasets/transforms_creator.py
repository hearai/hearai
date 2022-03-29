from typing import Dict

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class TransformsCreator:
    """
    Class for transforms creation and parametrization.
    """

    def __init__(self, augmentations_parameters: Dict = None):
        """
        Args:
            augmentations_parameters (Dict): Dict containing parameters regarding currently used augmentations.
                [Warning] Must contain fields:
                    - "apply_resize" (bool)
                    - "apply_center_crop" (bool)
                    - "apply_random_erasing" (bool)
                    - "apply_random_rotation" (bool)
                    - "apply_color_jitter" (bool)

                    - "resize_size" (int)
                    - "center_crop_size" (int)
                    - "random_erasing_probability" (float)
                    - "random_rotation_degree" (int)
                    - "color_jitter_brightness" (float)
                    - "color_jitter_contrast" (float)
                    - "color_jitter_saturation" (float)
                    - "color_jitter_brcolor_jitter_hueightness" (float)
        """

        self.apply_resize = augmentations_parameters["apply_resize"]
        self.apply_center_crop = augmentations_parameters["apply_center_crop"]
        self.apply_random_rotation = augmentations_parameters["apply_random_rotation"]
        self.apply_color_jitter = augmentations_parameters["apply_color_jitter"]
        self.apply_rgb_shift = True
        self.apply_blur = True

        self.resize_size = augmentations_parameters["resize_size"]

        self.center_crop_size = augmentations_parameters["center_crop_size"]

        self.random_rotation_degree = augmentations_parameters["random_rotation_degree"]

        self.color_jitter_brightness = augmentations_parameters["color_jitter_brightness"]
        self.color_jitter_contrast = augmentations_parameters["color_jitter_contrast"]
        self.color_jitter_saturation = augmentations_parameters["color_jitter_saturation"]
        self.color_jitter_hue = augmentations_parameters["color_jitter_hue"]

    def get_train_transforms(self) -> A.Compose:
        additional_augmentations = []

        if self.apply_random_rotation:
            additional_augmentations.append(A.Rotate(limit=self.random_rotation_degree))

        if self.apply_color_jitter:
            additional_augmentations.append(A.ColorJitter(brightness=self.color_jitter_brightness,
                                                          contrast=self.color_jitter_contrast,
                                                          saturation=self.color_jitter_saturation,
                                                          hue=self.color_jitter_hue))

        if self.apply_rgb_shift:
            additional_augmentations.append(A.RGBShift(r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2))

        if self.apply_blur:
            additional_augmentations.append(A.Blur(blur_limit=1))

        return self._get_transforms(additional_augmentations)

    def get_val_transforms(self) -> A.Compose:
        additional_augmentations = []
        return self._get_transforms(additional_augmentations)

    def _get_transforms(self, additional_augmentations: list) -> A.Compose:
        return A.Compose(
            [
                *additional_augmentations,
                A.SmallestMaxSize(max_size=self.resize_size),  # image batch, resize smaller edge to 256
                A.CenterCrop(width=self.center_crop_size, height=self.center_crop_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1),
                ToTensorV2()
            ]
        )

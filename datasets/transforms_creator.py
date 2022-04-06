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
                    - "apply_random_rotation" (bool)
                    - "apply_color_jitter" (bool)
                    - "apply_rgb_shift" (bool)
                    - "apply_blur" (bool)

                    - "resize_size" (int)
                    - "center_crop_size" (int)
                    - "random_erasing_probability" (float)
                    - "random_rotation_degree" (int)
                    - "color_jitter_brightness" (float)
                    - "color_jitter_contrast" (float)
                    - "color_jitter_saturation" (float)
                    - "color_jitter_brcolor_jitter_hueightness" (float)
        """

        self.apply_random_rotation = augmentations_parameters["apply_random_rotation"]
        self.apply_color_jitter = augmentations_parameters["apply_color_jitter"]
        self.apply_rgb_shift = augmentations_parameters["apply_rgb_shift"]
        self.apply_blur = augmentations_parameters["apply_blur"]

        self.resize_size = augmentations_parameters["resize_size"]
        self.center_crop_size = augmentations_parameters["center_crop_size"]

        self.random_rotation_degree = augmentations_parameters["random_rotation_degree"]
        self.color_jitter_brightness = augmentations_parameters["color_jitter_brightness"]
        self.color_jitter_contrast = augmentations_parameters["color_jitter_contrast"]
        self.color_jitter_saturation = augmentations_parameters["color_jitter_saturation"]
        self.color_jitter_hue = augmentations_parameters["color_jitter_hue"]
        self.rgb_shift_r_shift_limit = augmentations_parameters["rgb_shift_r_shift_limit"]
        self.rgb_shift_g_shift_limit = augmentations_parameters["rgb_shift_g_shift_limit"]
        self.rgb_shift_b_shift_limit = augmentations_parameters["rgb_shift_b_shift_limit"]
        self.blur_limit = augmentations_parameters["blur_limit"]

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
            additional_augmentations.append(A.RGBShift(r_shift_limit=self.rgb_shift_r_shift_limit,
                                                       g_shift_limit=self.rgb_shift_g_shift_limit,
                                                       b_shift_limit=self.rgb_shift_b_shift_limit))

        if self.apply_blur:
            additional_augmentations.append(A.Blur(blur_limit=self.blur_limit))

        return self._get_transforms(additional_augmentations)

    def get_val_transforms(self) -> A.Compose:
        additional_augmentations = []
        return self._get_transforms(additional_augmentations)

    def _get_transforms(self, additional_augmentations: list) -> A.Compose:
        return A.Compose(
            [
                *additional_augmentations,
                A.SmallestMaxSize(max_size=self.resize_size),
                A.CenterCrop(width=self.center_crop_size, height=self.center_crop_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1),
                ToTensorV2()
            ]
        )

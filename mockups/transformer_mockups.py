from typing import Tuple

import torch
import numpy as np


def generate_mockup_input(
    batch_size: int, input_shape: Tuple, value_range: Tuple
) -> torch.Tensor:
    """
    Function creates PyTorch tensor of specified shape with values in provided range.
    Args:
        batch_size (int): Number of instances in a single batch.
        input_shape (Tuple): Shape of input for single data instance.
        value_range (Tuple): Range of value to generate random float values from.

    Returns:
        (torch.Tensor) Mockup data tensor.
    """
    return torch.Tensor(
        np.random.uniform(
            low=value_range[0], high=value_range[1], size=(batch_size,) + input_shape
        )
    )


def generate_mockup_output(batch_size: int, num_classes: int) -> torch.Tensor:
    """
    Function creates PyTorch tensor which imitates output for a classifier model.

    Args:
        batch_size (int): Number of instances in a batch.
        num_classes (int): Number of classes to predict.

    Returns:
        (torch.Tensor) Mockup classification output.
    """
    classes = np.random.randint(low=0, high=num_classes, size=batch_size)
    classes_one_hot = np.zeros((batch_size, num_classes))
    classes_one_hot[np.arange(batch_size), classes] = 1

    return torch.Tensor(classes_one_hot)

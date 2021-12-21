from typing import Tuple

import torch
import numpy as np


def generate_mockup_input(batch_size: int,
                          input_shape: Tuple,
                          value_range: Tuple) -> torch.Tensor:
    """
    Function create Pytorch tensor of specified shape with values in provided range.
    Args:
        batch_size (int): Number of instances in a single batch.
        input_shape (Tuple): Shape of input for single data instance.
        value_range (Tuple): Range of value to generate random float values from.

    Returns:
        (torch.Tensor) Mockup data tensor.
    """
    return torch.Tensor(np.random.uniform(low=value_range[0], high=value_range[1], size=(batch_size,) + input_shape))


def generate_mockup_output(batch_size: int,
                           num_classes: int) -> torch.Tensor:
    if batch_size > num_classes:
        raise('Unsupported feature.')
    else:
        return torch.Tensor(np.apply_along_axis(np.random.permutation, 1, np.eye(num_classes)[:batch_size]))


if __name__ == '__main__':
    # Example usage
    batch_size = 3
    input_shape = (1, 2, 3)
    value_range = (0, 1)
    num_classes = 10

    mockup_input = generate_mockup_input(batch_size=batch_size,
                                         input_shape=input_shape,
                                         value_range=value_range)
    print(mockup_input.shape)

    mockup_output = generate_mockup_output(batch_size=batch_size,
                                           num_classes=num_classes)

import itertools
from collections import namedtuple
from typing import List

from tqdm import tqdm


def select_n_largest_sums(model_output: List[List[float]], n: int):
    model_output = [[(idx, value) for idx, value in enumerate(head)] for head in model_output]
    heads_product = itertools.product(*model_output)

    HeadSum = namedtuple("HeadSum", "indices value")
    heads_sums = []

    for head_sum_raw in tqdm(heads_product):
        indices = [head_raw[0] for head_raw in head_sum_raw]
        value = sum([head_raw[1] for head_raw in head_sum_raw])

        head_sum = HeadSum(indices, value)
        heads_sums.append(head_sum)
        if len(heads_sums) > 10:
            break

    head_sums = sorted(heads_sums, key=lambda x: x.value, reverse=True)[:n]


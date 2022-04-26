import json
from typing import List, Dict


def calc_distance(heads_pred: List[int], heads_test: List[int]) -> int:
    """
    Calculates distance between two lists of heads classes.
    If classes are the same 0 is assigned, 1 otherwise. Returned distance is a sum of these values.
    Args:
        heads_pred: list of predicted head classes
        heads_test: list of considered head classes for one gloss from the head_to_word_dict
    Returns:
        distance: numeric value describing the distance between heads_pred and heads_test, the less the better
    """

    distance = 0
    for head_test, head_pred in zip(heads_test, heads_pred):
        distance += head_test != head_pred
    return distance


def translate_heads(heads_pred: List[int], head_to_word_dict: Dict) -> (int, List[str]):
    """
    Translates predicted heads classes into the understandable gloss.
    Args:
        heads_pred: list of predicted head classes
        head_to_word_dict: dict which maps idx into its gloss and heads representation,
        the distance from considered heads_pred is included,
        e.g.

        1234: {
            "gloss": "test",
            "heads": [1,2,3,4,5,6,7,8],
            "distance": 0
        }
    Returns:
        min_distance - numerical value of the smallest distance in head_to_word_dict
        gloss_translations - list of translations with the min_distance
    """

    for idx in head_to_word_dict.keys():
        head_to_word_dict[idx]["distance"] = calc_distance(heads_pred, head_to_word_dict[idx]["heads"])

    distances = [head_to_word_dict[idx]["distance"] for idx in head_to_word_dict.keys()]
    min_distance = min(distances)

    gloss_translations = [head_to_word_dict[idx]["gloss"] for idx in head_to_word_dict.keys() if
                          head_to_word_dict[idx]["distance"] == min_distance]

    return min_distance, gloss_translations


def load_hamnosys_anns(hamnosys_anns_path: str) -> Dict:
    """
    Loads hamnosys anns from the hamnosys anns file.
    Args:
        hamnosys_anns_path - path to the source of the fully hamnosys annotations file
    Returns:
        hamnosys_anns - dictionary which maps gloss id to its heads classes configuration
    """
    hamnosys_anns = {}
    with open(hamnosys_anns_path) as anns_file:
        for line in anns_file.readlines()[1:]:
            line = line.replace('\n', '')
            line = [int(sign) for sign in line.split(' ') if sign]
            name = line[0] - 1
            heads = line[3:]

            hamnosys_anns[name] = heads
    return hamnosys_anns


def load_hamnosys_anns_dict(hamnosys_anns_dicts_path: str) -> Dict:
    """
    Loads hamnosys anns dict from the hamnosys anns dict file.
    Args:
        hamnosys_anns_dicts_path - path to the source of the fully hamnosys dict annotations file
    Returns:
        hamnosys_anns_dict - dictionary which maps gloss id to its translation
    """

    with open(hamnosys_anns_dicts_path) as anns_dict_file:
        anns_dict_file = anns_dict_file.read()
        hamnosys_anns_dict = json.loads(anns_dict_file)

    return hamnosys_anns_dict


def load_head_to_word_dict(hamnosys_anns_path: str, hamnosys_anns_dicts_path: str) -> Dict:
    """
    Loads hamnosys anns from the hamnosys anns and hamnosys anns dict file.
    Args:
        hamnosys_anns_path - path to the source of the fully hamnosys annotations file
        hamnosys_anns_dicts_path - path to the source of the fully hamnosys dict annotations file
    Returns:
        hamnosys_anns - dictionary which maps gloss id to its translation and heads representation
    """

    hamnosys_anns_cut = load_hamnosys_anns(hamnosys_anns_path)
    hamnosys_anns_dict = load_hamnosys_anns_dict(hamnosys_anns_dicts_path)
    hamnosys_anns = {}

    for idx in hamnosys_anns_cut.keys():
        gloss = [gloss for gloss, idx_ in hamnosys_anns_dict.items() if idx_ == idx]
        if not gloss:
            print(f'There is no translation for the annotation {idx}')
            continue

        hamnosys_anns[idx] = {
            "gloss": gloss[0],
            "heads": hamnosys_anns_cut[idx],
            "distance": 0
        }

    return hamnosys_anns

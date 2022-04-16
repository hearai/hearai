import argparse

import torch
import yaml

from datasets.dataset_creator import DatasetCreator
from datasets.transforms_creator import TransformsCreator
from models.model import GlossTranslationModel


def analyse(model_config_path: str, model_ckpt_path: str):
    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    transforms_creator = TransformsCreator(model_config["augmentations_parameters"])
    dataset_creator = DatasetCreator(
        data_paths=model_config["general_parameters"]["data_paths"],
        classification_mode=model_config["train_parameters"]["classification_mode"],
        classification_heads=model_config["heads"][model_config["train_parameters"]["classification_mode"]],
        num_segments=model_config["train_parameters"]["num_segments"],
        time=model_config["train_parameters"]["time"],
        use_frames=model_config["train_parameters"]["use_frames"],
        use_landmarks=model_config["train_parameters"]["use_landmarks"],
        ratio=model_config["general_parameters"]["ratio_train_test"],
        pre_training=model_config["train_parameters"]["pre_training"],
        transforms_creator=transforms_creator)

    test_dataset = dataset_creator.get_test_subset()

    model = GlossTranslationModel.load_from_checkpoint(model_ckpt_path)
    model.eval()

    with torch.no_grad():
        output = model(test_dataset)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--input-dir-path", type=str)
    parser.add_argument("--output-dir-path", type=str)
    return parser


if __name__ == '__main__':
    ...

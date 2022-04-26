import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import PadCollate
from datasets.dataset_creator import DatasetCreator
from datasets.transforms_creator import TransformsCreator
from models.model import GlossTranslationModel
from utils import load_head_to_word_dict, translate_heads


def analyse(model_config_path: str, model_path: str, input_dir_path: str, hamnosys_anns_path: str,
            hamnosys_anns_dict_path: str):
    """
    Function to analyse the input frames and translate it to the understandable gloss.
    """

    with open(model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    transforms_creator = TransformsCreator(model_config["augmentations_parameters"])

    dataset_creator = DatasetCreator(
        data_paths=input_dir_path,
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
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=model_config['train_parameters']['batch_size'],
        num_workers=model_config['general_parameters']['workers'],
        collate_fn=PadCollate(total_length=model_config['train_parameters']['num_segments']),
        drop_last=False,
    )

    model = GlossTranslationModel(
        general_parameters=model_config["general_parameters"],
        train_parameters=model_config["train_parameters"],
        feature_extractor_parameters=model_config["feature_extractor"],
        transformer_parameters=model_config["transformer"],
        heads=model_config["heads"],
        freeze_scheduler=model_config["freeze_scheduler"])

    model.load_state_dict(torch.load(model_path))
    model.eval()

    head_to_word_dict = load_head_to_word_dict(hamnosys_anns_path, hamnosys_anns_dict_path)

    for frames, landmarks, _ in test_dataloader:
        with torch.no_grad():
            output = model((frames, landmarks))
            output = [class_output.cpu().detach().numpy().tolist()[0] for class_output in output]
            heads = [np.argmax(class_output) for class_output in output]
            min_distance, gloss_translation = translate_heads(heads, head_to_word_dict)
            print(f'The most probable gloss translation (min distance: {min_distance}) are: {gloss_translation}')


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="Path to saved model in pth format. "
                             "Architecture should be defined in the model config file.")
    parser.add_argument("--model_config_path", type=str,
                        help="Path to the model config file in yml format. "
                             "The same structure as model config for training is needed.")
    parser.add_argument("--input_dir_path", type=str,
                        help="Path to the directory with input for the analysis. "
                             "It should contain subdirs with frames and annotation file in txt form.")
    parser.add_argument("--hamnosys_anns_path", type=str,
                        help="Path to the source of the fully hamnosys annotations file.")
    parser.add_argument("--hamnosys_anns_dict_path", type=str,
                        help="Path to the source of the fully hamnosys annotations dict file.")
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    analyse(args.model_config_path, args.model_path, args.hamnosys_anns_path, args.hamnosys_anns_dict_path)

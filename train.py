import argparse
import os
import random
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import PadCollate
from datasets.dataset_creator import DatasetCreator
from datasets.transforms_creator import TransformsCreator
from models.model import GlossTranslationModel
from models.model_for_pretraining import PreTrainingModel

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        type=str,
        default='train_config.yml',
        help="path to .yaml config file specifying hyperparameters of different model sections."
    )
    return parser


def main(args):
    with open(args.model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    # set GPU to use
    if model_config['general_parameters']['gpu'] > -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config['general_parameters']['gpu'])
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # set the seed for reproducibility
    seed = model_config['general_parameters']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.model_config_path) as file:
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

    train_subset, val_subset = dataset_creator.get_train_and_val_subsets()

    # prepare dataloaders
    dataloader_train = DataLoader(
        train_subset,
        shuffle=True,
        batch_size=model_config['train_parameters']['batch_size'],
        num_workers=model_config['general_parameters']['workers'],
        collate_fn=PadCollate(total_length=model_config['train_parameters']['num_segments']),
        drop_last=False,
    )

    dataloader_val = DataLoader(
        val_subset,
        shuffle=False,
        batch_size=model_config['train_parameters']['batch_size'],
        num_workers=model_config['general_parameters']['workers'],
        collate_fn=PadCollate(total_length=model_config['train_parameters']['num_segments']),
        drop_last=False,
    )

    # prepare model
    if model_config['train_parameters']['pre_training']:
        model_instance = PreTrainingModel
    else:
        model_instance = GlossTranslationModel

    model = model_instance(
        general_parameters=model_config["general_parameters"],
        train_parameters=model_config["train_parameters"],
        feature_extractor_parameters=model_config["feature_extractor"],
        transformer_parameters=model_config["transformer"],
        heads=model_config["heads"],
        freeze_scheduler=model_config["freeze_scheduler"],
        steps_per_epoch=max(1, len(train_subset) // model_config["train_parameters"]["batch_size"]))

    trainer = pl.Trainer(
        max_epochs=model_config['train_parameters']['epochs'],
        val_check_interval=1.0,
        gpus=[0] if model_config['general_parameters']['gpu'] > -1 else None,
        progress_bar_refresh_rate=10,
        accumulate_grad_batches=1,
        fast_dev_run=model_config['train_parameters']['fast_dev_run'],
    )

    # run training
    trainer.fit(
        model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val
    )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    
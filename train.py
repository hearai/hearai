import yaml
import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
import numpy as np
import random
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, random_split

from datasets.dataset import ImglistToTensor, PadCollate, VideoFrameDataset
from models.model_for_pretraining import PreTrainingModel
from models.model import GlossTranslationModel

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

    # trasformations
    preprocess = T.Compose(
        [
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            T.Resize(256),  # image batch, resize smaller edge to 256
            T.CenterCrop(256),  # image batch, center crop to square 256x256
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load data
    videos_root = model_config['general_parameters']['data_path']
    if not isinstance(videos_root, list):
        videos_root = [videos_root]

    datasets = list()
    for video_root in videos_root:
        if model_config['train_parameters']['classification_mode'] == "gloss":
            annotation_file = os.path.join(video_root, "test_gloss.txt")
        elif "hamnosys" in model_config['train_parameters']['classification_mode']:
            annotation_file = os.path.join(video_root, "test_hamnosys.txt")

        dataset = VideoFrameDataset(
            root_path=video_root,
            annotationfile_path=annotation_file,
            classification_heads = model_config["heads"][model_config['train_parameters']['classification_mode']],
            is_pretraining=model_config['train_parameters']['pre_training'],
            num_segments=model_config['train_parameters']['num_segments'],
            time=model_config['train_parameters']['time'],
            landmarks=model_config['train_parameters']['landmarks'],
            transform=preprocess,
            test_mode=True,
        )
        datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)

    # split into train/val
    train_val_ratio = model_config['general_parameters']['ratio_train_test']
    train_len = round(len(dataset) * train_val_ratio)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset, [train_len, val_len])

    # prepare dataloaders
    dataloader_train = DataLoader(
        train,
        shuffle=True,
        batch_size=model_config['train_parameters']['batch_size'],
        num_workers=model_config['general_parameters']['workers'],
        collate_fn=PadCollate(total_length=model_config['train_parameters']['num_segments']),
        drop_last=False,
    )

    dataloader_val = DataLoader(
        val,
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
        freeze_scheduler=model_config["freeze_scheduler"])

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

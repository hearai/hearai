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
from models.model import GlossTranslationModel
from models.model_for_pretraining import PreTrainingModel

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Data parameters and paths
    parser.add_argument(
        "--data", help="path to data", nargs="*", default=["assets/sanity_check_data"],
    )
    parser.add_argument(
        "--landmarks",
        action="store_true",
        default=False,
        help="flag to enable reading landmarks annotations",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default='train_config_default.yml',
        help="path to .yaml config file specyfing hyperparameters of different model sections."
    )
    parser.add_argument(
        "--classification-mode",
        default="hamnosys",
        choices=["gloss", "hamnosys", "hamnosys-less"],
        help="mode for classification, choose from classification_mode.py",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="train/test ratio (default: 0.8)"
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=8,
        help="dataset parameter defining number of segments per video used as an input",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="dataset parameter defining time to collect frames per video used as an input",
    )
    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=3,
        help="input batch size for training (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="EPOCHS",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of parallel workers (default: 0)",
    )
    # Other
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU number to use, run on CPU if default (-1) used",
    )
    parser.add_argument("--save", help="path to save model", default="./best.pth")
    parser.add_argument(
        "--seed", type=int, default=2021, help="random seed (default: 2021)"
    )
    parser.add_argument(
        "--fast-dev-run",
        default=False,
        action="store_true",
        help="Flag for a sanity-check, runs single loop for the training phase",
    )
    parser.add_argument(
        "--pre-training",
        default=False,
        action="store_true",
        help="Flag for a pre-training. Enables feature-extractor pre-training.",
    )
    # Neptune settings
    parser.add_argument(
        "--neptune",
        action="store_true",
        default=False,
        help="Launch experiment and log metrics with neptune",
    )
    return parser


def main(args):
    # set GPU to use
    if args.gpu > -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # set the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.model_config_path) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_creator = DatasetCreator(args.data, args.classification_mode,
                                     model_config["heads"][args.classification_mode], args.num_segments, args.time,
                                     args.landmarks, args.ratio, args.pre_training)

    train_subset, val_subset = dataset_creator.get_train_and_val_subsets()

    # prepare dataloaders
    dataloader_train = DataLoader(
        train_subset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=PadCollate(total_length=args.num_segments),
        drop_last=False,
    )

    dataloader_val = DataLoader(
        val_subset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=PadCollate(total_length=args.num_segments),
        drop_last=False,
    )

    # prepare model
    if args.pre_training:
        model_instance = PreTrainingModel
    else:
        model_instance = GlossTranslationModel

    model = model_instance(lr=args.lr,
                           classification_mode=args.classification_mode,
                           classification_heads=model_config["heads"][args.classification_mode],
                           feature_extractor_name="cnn_extractor",
                           feature_extractor_model_path=model_config["feature_extractor"]["model_path"],
                           transformer_name="sign_language_transformer",
                           num_attention_heads=model_config["transformer"]["num_attention_heads"],
                           transformer_dropout_rate=model_config["transformer"]["dropout_rate"],
                           num_segments=args.num_segments,
                           model_save_dir=args.save,
                           neptune=args.neptune,
                           representation_size=model_config["feature_extractor"]["representation_size"],
                           feedforward_size=model_config["transformer"]["feedforward_size"],
                           num_encoder_layers=model_config["transformer"]["num_encoder_layers"],
                           transformer_output_size=model_config["transformer"]["output_size"],
                           warmup_steps=20.0,
                           multiply_lr_step=0.95,
                           freeze_scheduler=model_config["freeze_scheduler"]
                           )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=1.0,
        gpus=[0] if args.gpu > -1 else None,
        progress_bar_refresh_rate=10,
        accumulate_grad_batches=1,
        fast_dev_run=args.fast_dev_run,
    )

    # run training
    trainer.fit(
        model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val
    )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

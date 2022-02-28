import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
import np
import random
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, random_split

from datasets.dataset import ImglistToTensor, PadCollate, VideoFrameDataset
from models.model import GlossTranslationModel

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Data parameters and paths
    parser.add_argument(
        "--data", help="path to data", default="assets/sanity_check_data",
    )
    parser.add_argument(
        "--landmarks_path",
        type=str,
        default=None,
        help="path to landmarks annotations",
    )
    parser.add_argument(
        "--classification-mode",
        default="hamnosys",
        choices=["gloss", "hamnosys"],
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
    if args.gpu > 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # set the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load data
    videos_root = args.data
    if args.classification_mode == "gloss":
        annotation_file = os.path.join(videos_root, "test_gloss.txt")
    elif args.classification_mode == "hamnosys":
        annotation_file = os.path.join(videos_root, "test_hamnosys.txt")

    preprocess = T.Compose(
        [
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            T.Resize(256),  # image batch, resize smaller edge to 256
            T.CenterCrop(256),  # image batch, center crop to square 256x256
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        classification_mode=args.classification_mode,
        num_segments=args.num_segments,
        time=args.time,
        landmarks_path=args.landmarks_path,
        transform=preprocess,
        test_mode=False,
    )

    # split into train/val
    train_val_ratio = args.ratio
    train_len = round(len(dataset) * train_val_ratio)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset, [train_len, val_len])

    # prepare dataloaders
    dataloader_train = DataLoader(
        train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=PadCollate(total_length=args.num_segments),
        drop_last=False,
    )

    dataloader_val = DataLoader(
        val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=PadCollate(total_length=args.num_segments),
        drop_last=False,
    )

    # prepare model
    model = GlossTranslationModel(
        lr=args.lr,
        classification_mode=args.classification_mode,
        feature_extractor_name="cnn_extractor",
        feature_extractor_model_path="efficientnet_b2",
        transformer_name="sign_language_transformer",
        num_attention_heads=4,
        num_segments=args.num_segments,
        model_save_dir=args.save,
        neptune=args.neptune,
        device="cuda:0" if args.gpu > 0 else "cpu",
        representation_size=512,
        feedforward_size=1024,
        transformer_output_size=784,
        warmup_steps=20.0,
        multiply_lr_step=0.95,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=args.ratio,
        gpus=args.gpu if args.gpu > 0 else None,
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

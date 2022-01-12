import argparse
import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import yaml
from datasets.dataset import ImglistToTensor, VideoFrameDataset
from models.model import GlossTranslationModel
from torch.utils.data import DataLoader, Dataset, random_split


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Data parameters and paths
    parser.add_argument(
        "--data", help="path to data", default="/dih4/dih4_2/hearai/data/frames/pjm"
    )
    parser.add_argument("--classes", type=int, default=2400, help="number of classes")
    parser.add_argument(
        "--ratio", type=float, default=0.9, help="train/test ratio (default: 0.9)"
    )
    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="EPOCHS",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of parallel workers (default: 0)"
    )
    # Other
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="number of GPU to use, run on CPU if default (-1) used",
    )
    parser.add_argument("--save", help="path to save model", default="./best.pth")
    parser.add_argument(
        "--seed", type=int, default=2021, help="random seed (default: 2021)"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Flag for a sanity-check, runs single loop for the training phase",
    )
    return parser


def main(args):
    # set GPU to use
    if args.gpu > 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # set torch seed
    torch.manual_seed(args.seed)

    # basic transforms
    test_transforms = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])

    # load data
    videos_root = args.data
    annotation_file = os.path.join(videos_root, "test_gloss.txt")
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
        num_segments=1,
        frames_per_segment=1,
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
        drop_last=False,
    )

    dataloader_val = DataLoader(
        val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
    )

    # prepare model
    model = GlossTranslationModel(
        lr=args.lr,
        num_classes=args.classes,
        feature_extractor_name="cnn_extractor",
        model_save_dir=args.save,
    )

    # TODO - create NeptuneLogger

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=0.3,
        gpus=args.gpu if args.gpu > 0 else None,
        progress_bar_refresh_rate=20,
        accumulate_grad_batches=1,
        fast_dev_run=args.fast_dev_run,
    )

    # run training
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    # parser = get_args_parser()
    # args = parser.parse_args()
    # main(args)
    print('Hello github actions!')


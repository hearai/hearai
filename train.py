import os
import argparse
import torch
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.loggers import NeptuneLogger
from datasets.dataset import CustomDataset
from models.model import GlossTranslationModel

def get_args_parser():
    parser = argparse.ArgumentParser()
    # Data parameters and paths
    parser.add_argument('--data',
                        help='path to data')
    # Training parameters
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate (default: 3e-5)')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='EPOCHS',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of parallel workers (default: 0)')
    # Other
    parser.add_argument('--gpu', type=int, default=0,
                        help='number of GPU to use')
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed (default: 2021)')
    return parser


def main(args):
    # set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # set torch seed
    torch.manual_seed(args.seed)

    # load data
    dataset = CustomDataset(args.data)
    
    
    # split into train/val
    train_val_ratio = 0.9
    train_len = round(len(dataset)*train_val_ratio)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset, [train_len, val_len])
    
    
    # prepare dataloaders
    dataloader_train = DataLoader(train, shuffle=True, batch_size=args.batch_size,
                                        num_workers=args.workers, drop_last=False)
        
    dataloader_val = DataLoader(val, shuffle=False, batch_size=args.batch_size,
                                num_workers=args.workers, drop_last=False)

    # prepare model
    model = GlossTranslationModel(lr=args.lr,
                                  feature_extractor_path="cnn_extractor")

    
    # create NeptuneLogger
    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",  # replace with your own
        project="common/pytorch-lightning-integration",  # "<WORKSPACE/PROJECT>"
        tags=["training", "test"],  # optional
    )

    # pass it to the Trainer
    trainer = pl.Trainer(max_epochs=args.epochs, 
                    val_check_interval=0.3,
                    gpus=[0], 
                    progress_bar_refresh_rate=20,
                    logger=neptune_logger,
                    callbacks=[lr_monitor],
                    accumulate_grad_batches=1)

    # run training
    trainer.fit(model, dataloader_train, dataloader_val)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

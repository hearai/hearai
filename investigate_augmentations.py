import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from datasets.dataset import PadCollate
from datasets.dataset_creator import DatasetCreator
from datasets.transforms_creator import TransformsCreator


def get_dataloader_train():
    with open("train_config_default.yml") as file:
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

    train_subset, _ = dataset_creator.get_train_and_val_subsets()

    dataloader_train = DataLoader(
        train_subset,
        shuffle=True,
        batch_size=model_config["train_parameters"]["batch_size"],
        num_workers=1,
        collate_fn=PadCollate(total_length=model_config["train_parameters"]["num_segments"]),
        drop_last=False,
    )

    return dataloader_train


def investigate_augmentations():
    dataloader_train = get_dataloader_train()

    for imgs, _ in enumerate(dataloader_train):
        imgs = imgs[0].detach().numpy().transpose(0, 2, 3, 1)
        _, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
        plt.savefig('aug.png')
        break


if __name__ == '__main__':
    investigate_augmentations()

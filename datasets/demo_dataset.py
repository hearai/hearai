"""
This code was adopted from
https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
"""

from dataset import VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str, save: str):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
        axes_pad=0.3,  # pad between axes in inch.
    )
    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.savefig(save)


if __name__ == "__main__":
    """
    TABLE OF CODE CONTENTS:
    1. Minimal demo without image transforms
    2. Minimal demo without sparse temporal sampling for single continuous frame clips, without image transforms
    3. Demo with image transforms
    """

    videos_root = os.path.join("/dih4/dih4_2/hearai/data/frames/", "pjm")
    annotation_file = os.path.join(videos_root, "annotations.txt")

    """ DEMO 1 WITHOUT IMAGE TRANSFORMS """
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        transform=None,
        test_mode=False,
    )

    sample = dataset[0]
    frames = sample[0]  # list of PIL images
    label = sample[1]  # list of label/-s

    plot_video(
        rows=1,
        cols=5,
        frame_list=frames,
        plot_width=15.0,
        plot_height=3.0,
        title="Evenly Sampled Frames, No Video Transform",
        save="out-1.jpg",
    )

    """ DEMO 2 SINGLE CONTINUOUS FRAME CLIP INSTEAD OF SAMPLED FRAMES, WITHOUT TRANSFORMS """
    # If you do not want to use sparse temporal sampling, and instead
    # want to just load N consecutive frames starting from a random
    # start index, this is easy. Simply set NUM_SEGMENTS=1 and
    # FRAMES_PER_SEGMENT=N. Each time a sample is loaded, N
    # frames will be loaded from a new random start index.
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=1,
        frames_per_segment=9,
        transform=None,
        test_mode=False,
    )

    sample = dataset[3]
    frames = sample[0]  # list of PIL images
    label = sample[1]  # list of label/-s

    plot_video(
        rows=3,
        cols=3,
        frame_list=frames,
        plot_width=10.0,
        plot_height=5.0,
        title="Continuous Sampled Frame Clip, No Video Transform",
        save="out-2.jpg",
    )

    """ DEMO 3 WITH TRANSFORMS """
    # As of torchvision 0.8.0, torchvision transforms support batches of images
    # of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
    # transformations on the batch identically on all images of the batch. Any torchvision
    # transform for image augmentation can thus also be used  for video augmentation.
    preprocess = transforms.Compose(
        [
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.Resize(299),  # image batch, resize smaller edge to 299
            transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        transform=preprocess,
        test_mode=False,
    )

    sample = dataset[2]
    frame_tensor = sample[
        0
    ]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
    label = sample[1]  # list of label/-s

    print("Video Tensor Size:", frame_tensor.size())

    def denormalize(video_tensor):
        """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        """
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        return (
            (inverse_normalize(video_tensor) * 255.0)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )

    frame_tensor = denormalize(frame_tensor)
    plot_video(
        rows=1,
        cols=5,
        frame_list=frame_tensor,
        plot_width=15.0,
        plot_height=3.0,
        title="Evenly Sampled Frames, + Video Transform",
        save="out-3.jpg",
    )

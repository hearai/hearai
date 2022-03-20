![CI workflow badge](https://github.com/hearai/hearai/workflows/CI-pipeline/badge.svg) ![Visits Badge](https://badges.pufler.dev/visits/hearai/hearai)

<p align="center">
<a href="https://www.hearai.pl"><img src="https://i.imgur.com/wKCpSOh.png" height="auto" width="200"></a>
</p>

# HearAI
Model Sign Language with Machine Learning.

Deaf people are affected by many forms of exclusion, especially now in the pandemic world. [HearAI](https://www.hearai.pl/) aims to build a deep learning solution to make the world more accessible for the Deaf community and increase the existing knowledge base in using AI for Polish Sign Language.

# 🤖 Pipeline prototype
This is a prototypical repository of the HearAI project. Be careful as the work is still in progress!

## 💡 Idea

Video --> Frames --> Feature extractor --> Transformer --> Classification heads --> Prediction for video


## 🧪 Feature extractors
A feature extractor is a model used to extract features directly from the input: e.g., a set of video frames. For instance, we use a CNN that extracts features from each video frame with `multi_frame_feature_extractor`. Another approach could be extracting the pose of a signer and passing the coordinates to GCN (Graph CNN), for each frame separately. 
Feature extractor returns a representation feature vector for every frame.

## 🧠 Transformer
The transformer is a widely-used deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in natural language processing (NLP). The Transfomer model will get representation from feature_extractor of size `num_frames,representation_size` for each video in our pipeline.

## 📑 [WiP] Datasets
In our studies we are using PJM lexicon with annotations provided at our internal server at:
- directories with frames: `/dih4/dih4_2/hearai/data/frames/pjm/`
- HamNoSys annotation file: `/dih4/dih4_2/hearai/data/frames/pjm/test_hamnosys.txt` - 8 heads only [WiP]
- glosses annotation file: `/dih4/dih4_2/hearai/data/frames/pjm/test_gloss2.txt`

We have possibility to load these datasets:
- pjm: /dih4/dih4_2/hearai/data/frames/pjm
- basic_lexicon: /dih4/dih4_2/hearai/data/frames/basic_lexicon
- galex: /dih4/dih4_2/hearai/data/frames/galex
- glex: /dih4/dih4_2/hearai/data/frames/glex
- gsll: /dih4/dih4_2/hearai/data/frames/gsll_done

Attention! The method requires that all ```annotation_files``` have exactly the same annotation filename e.g. ```"test_hamnosys.txt"```! If you need to pass different path you need to do it manually.

For this file, you can create basic charts with statistics. Every chart contains information on how many times, every HamNoSys sign in a specific category occurs in the dataset.
As input arguments, you must pass a path to the file with annotations, and an output directory for generated charts.
Optionally, you can also pass a separator, which is used in the annotation file.
This script is named ```make_statistics``` and is in the ```utils``` directory.

## 👥 Classification heads
Pipeline handle multihead classification. We predefine `classification_heads` for both Gloss Translation and HamNoSys recognition. Our `classification_heads` are defined here: `utils/classification_mode.py`. For each head, a custom loss weight can be provided.

Hamburg Sign Language Notation System (HamNoSys) is a gesture transcription alphabetic system that describes the symbols and gestures such as hand shape, hand location, and movement. Read more about HamNoSys [here - Introduction to HamNoSys](https://www.hearai.pl/post/4-hamnosys/) and [here - Introduction to HamNoSys Part 2](https://www.hearai.pl/post/5-hamnosys2/). HamNoSys always have the same number of possible classes.

```
        "symmetry_operator": {
            "num_class": 9,
            "loss_weight": 0,
        },
        "hand_shape_base_form": {
            "num_class": 12,
            "loss_weight": 1,
        },
        "hand_shape_thumb_position": {
            "num_class": 4,
            "loss_weight": 0,
        },
        "hand_shape_bending": {
            "num_class": 6,
            "loss_weight": 0,
        },
        "hand_position_finger_direction": {
            "num_class": 18,
            "loss_weight": 0,
        },
        "hand_position_palm_orientation": {
            "num_class": 8,
            "loss_weight": 0,
        },
        "hand_location_x": {
            "num_class": 5,
            "loss_weight": 0,
        },
        "hand_location_y": {
            "num_class": 37,
            "loss_weight": 0,
        },

```

Gloss is an annotation system that applies a label (a word) to the sign. Number glosses depend on a language and dataset. It is usually a bigger number as it must define as many words (glosses) as possible.

```
"gloss": {
                "num_class": 2400,
                "loss_weight": 1
            }
```

# 🛠 Environment setup

To create a reproducible environment, create a virtual environment using venv and requirements defined.
In terminal run:
`make venv`

When you install a new library, please add it to the list in `requirements.txt` file so we can avoid dependency conflicts or failed build.

# 🏁 Example train.py run

Run with a single dataset:

`python3 train.py --model_config_path train_config.yml`

train_config.yml file should define the following fields:

`general_parameters` - parameters that do not affect the training process directly
- `data_path` - (str) dir where input data is stored.
- `ratio_train_test` - (float) train/test split ratio.
- `workers` - (int) number of parallel workers (default: 0).
- `gpu` - (int) number of GPU to train the model on. If -1 CPU will be used.
- `path_to_save` - (str) path where model will be saved during/after training.
- `seed` - (int) random seed number
- `neptune` - (bool) if True training's results will be stored in neptune.ai

`augmentations_parameters` - parameters describing augmentation methods that will be used while loading the data  
- `apply_resize` - (bool) if True resizing operation will be applied.
- `apply_center_crop` - (bool) if True cropping operation will be applied.
- `apply_random_erasing` - (bool) if True random erasing operation will be applied.
- `apply_random_rotation`  - (bool) if True random rotation will be applied.
- `apply_color_jitter` - (bool) if True color jitter operation will be applied.

- `resize_size` - (int) size of input image after resizing.
- `center_crop_size` - (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
- `random_erasing_probability` - (float) probability that the random erasing operation will be performed.
- `random_rotation_degree` - (int) degrees for rotation operation.
- `color_jitter_brightness` - (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
- `color_jitter_contrast` - (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
- `color_jitter_saturation` - (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
- `color_jitter_hue` - (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

`train_parameters` - parameters that affect the training process  
- `classification_mode` - (str) type of training, currently available are "hamnosys", "gloss"
- `epochs` - (int) number of training epochs.
- `batch_size` - (int) number of datapoints in batch.
- `lr` - (float) learning rate.
- `num_segments` - (int) Number of consecutive frames that will be used as input.
- `fast_dev_run` - (bool) if True model will be run in faster-development mode.
- `pre_training` - (bool) if True model will be run in pre_training mode.
- `landmarks` - (bool) if True all coordinates for right hand, left hand, face, pose will be read.
- `time` - (float) time-distance (in seconds) between consecutive frames that will be used as input.
- `warmup_steps` - (float) number of steps with reduced learning rate before training with the proper rate.
- `multiply_lr_step` - (float) rate of change of learning rate after each step.

`feature_extractor` - parameters required by feature extractor class
- `name` - feature extractor name, currently available "cnn_extractor", "resnet50_extractor".
- `model_path` - (str) name of the model as provided in {timm} module, necessary if using "cnn_extrctor".
- `representation_size` - (int) Number of output featuresto be constructed by any feature extractor.

`transformer` - parameters required by transformer class
- `name` - (str) currently available names are "sign_language_transformer", "hubert_transformer", "fake_transformer".
- `feedforward_size` - (int) number of feedforward features in "sign_language_transformer".
- `num_attention_heads` - (int) number of attention heads in "sign_language_transformer".
- `num_encoder_layers` - (int) number of encoder layers in "sign_language_transformer".
- `output_size` - (int) output size of the final layer "sign_language_transformer", "hubert_transformer" & "fake_transformer".
- `dropout_rate` - (float) dropout rate in "sign_language_transformer".

`heads` - parameters defining output heads, number of available classes at each head and every head's weight
    Example - there are two available training choices 'gloss' & 'hamnosys' each with different number of heads.
- `heads`:
    `gloss`:
        `gloss`:
            `num_class`: 2400
            `loss_weight`: 1
    `hamnosys`:
        `symmetry_operator`:
            `num_class`: 9
            `loss_weight`: 0.125
        `hand_shape_base_form`:
            `num_class`: 12
            `loss_weight`: 0.125

`freeze_scheduler` - description of freezing/unfreezing of the feature extractor and tranformer during the training.
    Example - `freeze_scheduler` config is presented below. For this configuration, freeze_scheduler will freeze `feature_extractor` for ten epochs, and then- `freeze` `transformer` for five. After freezeing patterns ends, all params are unfreezed.
- `freeze_scheduler`:
    `model_params`:
        `feature_extractor`: [True, False]
        `transformer`: [False, True]
    `freeze_pattern_repeats`: [10, 5]
    `freeze_mode`: "epoch"
    `verbose`: True

# Training tips & tricks

### Schedule model freezing
If you use a pretrained `feature_extractor` but training `transfomer` from sctratch its worth to freeze weights of feature_extractor at first, and train the model with higher learning rate. With `freeze_scheduler` you can quickly prepare your freezing configuration. Freeze pattern can be used with any named model parameter. To use it, first select layers which you want to freeze, and add named parameters to `model_params` in the config file. For instance, it can be used with `feature_extractor` and `transformer`. When the value is set to `True` then the layer will be freezed after first freeze_scheduler execution. 

The `freeze_scheduler` config is explained below:

```python
"freeze_scheduler": {
    "model_params": {
        # each params list has to be of equal length
        "feature_extractor":  list(bool) # A list of bools defining freezing patterns for feature_extractor. True => Freeze. False => Train
        "transformer": list(bool) # A list of bools defining freezing patterns for transformer. True => Freeze. False => Train
    },
    "freeze_pattern_repeats": list(int), # A list of integers defines how many times each param in model_params will be repeated.
    "freeze_mode": str, # the freeze scheduler can be executed either every "epoch" or "step"
}

```

# 🎨 Style

For style formatting, the `black` library is used as default.
If you want to check whether your code matches style requirements, in the terminal run:
`make format-check`
If you want to apply changes to your code formatting style, in the terminal run:
`make format-apply`

To check code quality with linter, in the terminal run:
`make lint`
Currently, `pylint` is the default linter with access to train.py and models/ (all files are evaluated altogether, if you want to test your specific file, try: `pylint yourfile.py`)

# 🪐 How to setup logging with neptune.ai
- go to your neptune.ai account and get your API token
- in terminal, add your personal token to environmental variables
`export NEPTUNE_API_TOKEN = "<your_token>"`
- go to your neptune.ai account and get your project name
- in terminal, add your project name to environmental variables
`export NEPTUNE_PROJECT_NAME = "<your_workspace/your_project_name>"`
- if you want to make sure that your credentials are saved properly, you can use `printenv`
- to run training with Neptune logger initialized, add `--neptune` flag, i.e. `python3 train.py --neptune`


# 💻 Project organization
------------

    ├── LICENSE
    ├── README.md
    |         <- The top-level README for developers using this project.
    ├── data_preprocessing <- scripts for dataset preprocessing, e.g. to convert all datasets to the same format
    │   
    ├── datasets          <- scripts for data loading. One file per dataset
    │
    ├── models            <- scripts for PyTorch models. One file per model
    │
    ├────────── feature_extractors  <- scripts for feature extraction models: e.g. CNN or GCN. Takes a sequence of video frames or coordinates as inputs
    │
    ├────────── transformers        <- scripts for transformers models: e.g. BERT. Takes representation extracted by feature_extractors as input
    │
    ├── notebooks          <- jupyter notebooks.
    │   
    ├── utils              <- source code with useful functions
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`

------------


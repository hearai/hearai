![CI workflow badge](https://github.com/hearai/hearai/workflows/CI-pipeline/badge.svg) ![Visits Badge](https://badges.pufler.dev/visits/hearai/hearai)
# HearAI
Model Sign Language with Machine Learning.

Deaf people are affected by many forms of exclusion, especially now in the pandemic world. [HearAI](https://www.hearai.pl/) aims to build a deep learning solution to make the world more accessible for the Deaf community and increase the existing knowledge base in using AI for Polish Sign Language.

# Pipeline prototype
This is a prototypical repository o the HearAI project. Be careful as the work is still in progress!

## Idea

Video --> Frames --> Feature extractor --> Transformer --> Classification heads --> Prediction for video


## Feature extractors
A feature extractor is a model used to extract features directly from the input: e.g., a set of video frames. For instance, we use a CNN that extracts features from each video frame with `multi_frame_feature_extractor`. Another approach could be extracting the pose of a signer and passing the coordinates to GCN (Graph CNN), for each frame separately. 
Feature extractor returns a representation feature vector for every frame.

## Transformer
The transformer is a widely-used deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in natural language processing (NLP). The Transfomer model will get representation from feature_extractor of size `num_frames,representation_size` for each video in our pipeline.

## Classification heads
Pipeline handle multihead classification. We predefine `classification_heads` for both Gloss Translation and HamNoSys recognition. Our `classification_heads` are defined here: `utils/classification_mode.py`

Hamburg Sign Language Notation System (HamNoSys) is a gesture transcription alphabetic system that describes the symbols and gestures such as hand shape, hand location, and movement. Read more about HamNoSys [here - Introduction to HamNoSys](https://www.hearai.pl/post/4-hamnosys/) and [here - Introduction to HamNoSys Part 2](https://www.hearai.pl/post/5-hamnosys2/). HamNoSys always have the same number of possible classes.


```
"hand_shape_base_form": 6,
"hand_shape_thumb_position": 3,
"hand_shape_bending": 4,
"hand_position_finger_direction": 18,
"hand_position_palm_orientation": 8,
"hand_location_x": 14,
"hand_location_y": 5,
```

Gloss is an annotation system that applies a label (a word) to the sign. Number glosses depend on a language and dataset. It is usually a bigger number as it must define as many words (glosses) as possible.

```
"gloss": 2400
```

# Environment setup

To create a reproducible environment, create a virtual environment using venv and requirements defined.
In terminal run:
`make venv`

When you install a new library, please add it to the list in `requirements.txt` file so we can avoid dependency conflicts or failed build.

# Example train.py run

`python3 train.py --data /dih4/dih4_2/hearai/data/frames/pjm --gpu 1`


# Style

For style formatting, the `black` library is used as default.
If you want to check whether your code matches style requirements, in the terminal run:
`make format-check`
If you want to apply changes to your code formatting style, in the terminal run:
`make format-apply`

To check code quality with linter, in the terminal run:
`make lint`
Currently, `pylint` is the default linter with access to train.py and models/ (all files are evaluated altogether, if you want to test your specific file, try: `pylint yourfile.py`)

# How to setup logging with neptune.ai
- go to your neptune.ai account and get your API token
- in terminal, add your personal token to environmental variables
`export NEPTUNE_API_TOKEN = "<your_token>"`
- go to your neptune.ai account and get your project name
- in terminal, add your project name to environmental variables
`NEPTUNE_PROJECT_NAME = "<your_workspace/your_project_name>"`
- if you want to make sure that your credentials are saved properly, you can use `printenv`
- to run training with Neptune logger initialized, add `--neptune` flag, i.e. `python3 train.py --neptune`


# Project organization
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


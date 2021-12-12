# Example pipeline
This branch presents the pseudo-code of example pipeline.
Some parts of codes are ready-to-use, but others are just a dummy examples.
This is an idea to share. The final version of the repository might look differently, but this is a nice starter.

# Idea
We should be able to eaisly switch models used in the main model e.g. in the `GlossTranslationModel`, for instance, by giving a different path to a model
We want to eaisly switch datasets (e.g. just change import)
`train.py` and final model should be quite simple. Everything more complicated should be moved to submodels e.g. a transfomer model should not be implemented in the main model. Instead we would prefer to load it `models/transfomers/vanilla_transfomer`
Submodels should be reusable: key variables should be variables (e.g. `input_size`)
Main model is written in PyTorch Lightning for easier training. But for inference we will load it like a normal PyTorch model.

# Project organization
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
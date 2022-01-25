# How to run inference?

## MP4 input

```
python demo-mp4.py file.mp4 > result.csv
```

## Directory with jpgs

```
python demo-jpgs.py dir > result.csv
```



---------------------



Original code from: https://github.com/zmurez/MediaPipePyTorch.git

Original Readme file below:

# MediaPipe in PyTorch

Port of MediaPipe (https://github.com/google/mediapipe) tflite models to PyTorch

Builds upon the work of https://github.com/hollance/BlazeFace-PyTorch

```python demo.py```

## Models ported so far
1. Face detector (BlazeFace)
1. Face landmarks
1. Palm detector
1. Hand landmarks

## TODO
1. Add conversion and verification scripts
1. Verify face landmark pipeline
1. Improve README and samples

# Body landmark distance calculation script

This script can be used to calculate mean and stdev of L1 and L2
between 2 points on the body. The points on the body are detected
using Media Pipe Body Landmark model.

## Installation

Install required dependencies specified in the environment.yml file
or create a new conda environment from this config.

```bash
conda env create -f environment.yml
```

## Usage

In the image directory organize data in subdirectories,
where each subdirectory contains images corresponding to a specific
class as indicated by the HamNoSys notation.

Then, run the script
```bash
python test_pose_estimation.py --image_dir images
```
From the command line, provide two body landmarks between which
you want to calculate the distance as comma separated list, e.g.
1,2 meaning you want to select point 1 and 2

{0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer', 4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer', 7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right', 11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow', 15: 'left_wrist', 16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky', 19: 'left_index', 20: 'right_index', 21: 'left_thumb', 22: 'right_thumb', 23: 'left_hip', 24: 'right_hip', 25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle', 29: 'left_heel', 30: 'right_heel', 31: 'left_foot_index', 32: 'right_foot_index'}

Output will be provided in stdout as:
                        value
                         mean       std
category       norm
images/beard   l1    7.829410       NaN
               l2    5.650865       NaN
images/chin    l1    7.606648       NaN
               l2    5.663908       NaN
images/eyes    l1    6.928789  0.106352
               l2    5.078944  0.160248
images/head    l1    7.209588  0.540509
               l2    5.212561  0.473193
images/headtop l1    7.628600  0.363850
               l2    5.445853  0.248022

and as plots showing mean and stdev.

## License
[MIT](https://choosealicense.com/licenses/mit/)

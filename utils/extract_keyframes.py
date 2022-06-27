#code based on https://github.com/krxat/keyframe-detection/blob/master/keyframe.ipynb

import cv2
import os
import numpy as np
import argparse

    """
    Threshold values used in our project:
    basic lexicon : 30000
    pjm : 2180000
    glex/galex : 10000
    gsll: 3700000
    """
    
def get_args_parser():
    parser = argparse.ArgumentParser("Extract keyframes from frame sequence")
    parser.add_argument(
        "--path",
        metavar="DIR",
        help="path to base directory with frames",
      
    )
    
    parser.add_argument(
        "--dest",
        metavar="DEST",
        help="path to destination directory",
        
    )
    
    parser.add_argument(
        "--thr",
        metavar="N",
        type=int,
        help="path to destination directory",
        
    )
    
    return parser

def keyframeDetection(dir, frames_path, threshold, keyframePath):
    
    cap = cv2.VideoCapture(frames_path, cv2.CAP_IMAGES)
	    
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)
	    # Read the first frame.
    ret, prev_frame = cap.read()

    frame_nr = 0
    count = 0
       
    while ret:
        ret, curr_frame = cap.read()
        if ret:
                
            diff = cv2.absdiff(curr_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count > threshold:
                if not os.path.exists(keyframePath+str(dir)):
                    os.makedirs(keyframePath+str(dir))
                cv2.imwrite(keyframePath+str(dir) + "/" + dir + "_" + str(frame_nr) + '.jpg', curr_frame)
                
                frame_nr += 1
            count += 1
            prev_frame = curr_frame
            
    
            
    print("Total Number of frames saved: {}".format(count))



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    dir_path = args.path
    
    p,d,f = next(os.walk(dir_path))
    for dir in d:
        try:
	        keyframeDetection(dir,frames_path = dir_path + "{}/{}_%d.jpg".format(str(dir),str(dir)), 
                              threshold = args.thr, keyframePath = args.dest)
        except cv2.error:
            pass
        

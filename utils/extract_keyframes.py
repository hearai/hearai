#code based on https://github.com/krxat/keyframe-detection/blob/master/keyframe.ipynb

import cv2
import os
import numpy as np
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Extract keyframes from frame sequence")
    parser.add_argument(
        "--path",
        metavar="DIR",
        help="path to base directory with frames",
        default="/dih4/dih4_2/hearai/data/frames/gsll_done/",
    )
    
    parser.add_argument(
        "--dest",
        metavar="DIR",
        help="path to destination directory",
        default="/dih4/dih4_2/hearai/nczerep/gsll-frames",
    )
    
    parser.add_argument(
        "--thr",
        metavar="N",
        type=int,
        help="path to destination directory",
        default="10000"
    )
    
    return parser

def keyframeDetection(dir, frames_path, threshold, keyframePath):
    
    cap = cv2.VideoCapture(frames_path, cv2.CAP_IMAGES)
	    
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)
	    # Read the first frame.
    ret, prev_frame = cap.read()

    i = 0
    count = 0
            
    if not os.listdir(keyframePath+str(dir)+"/"):
        
        while ret:
            ret, curr_frame = cap.read()
            if ret:
                
                diff = cv2.absdiff(curr_frame, prev_frame)
                non_zero_count = np.count_nonzero(diff)
                if non_zero_count > threshold:
                    print("Saving Frame number: {}".format(i), end='\r')
                    cv2.imwrite(keyframePath+str(dir)+"/"+dir+"_"+str(i)+'.jpg',curr_frame)
                    i+=1
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
                              threshold = args.thr, keyframes_path = args.dest)
        except cv2.error:
            pass
        

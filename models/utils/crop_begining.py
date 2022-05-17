# based on: https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# and: https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42
import os
import argparse
import imutils
import cv2
import pandas as pd
import tqdm

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-ai", "--ann-input", type=str, help="path to source annotations")
    ap.add_argument("-ao", "--ann-output", type=str, help="path to output annotations")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    ap.add_argument("--jpg", action="store_true", default=False,
        help="Decide if you are reading from video")
    args = vars(ap.parse_args())
    # read annotations
    df = pd.read_csv(args["ann_input"], delimiter=" ")
    df.dropna(inplace=True, axis=1)
    # prepare variables
    start_new = []
    film_i = -1
    # loop over the all videos in annotation file
    for name in tqdm.tqdm(df.name.to_numpy()):
        num_i = -1
        film_i += 1
        # we are reading from a video file or just separate frames
        if not args["jpg"]:
            vs = cv2.VideoCapture(
                os.path.join(args["video"], str(name) + ".mp4"))
        # initialize the first frame in the video stream
        firstFrame = None
        # loop over the frames of the video
        succes = True
        # in some videos we can do not find any movement
        start_new.append(0)
        while succes:
            num_i += 1
            # grab the current frame
            if not args["jpg"]:
                frame = vs.read()[1]
            else:
                frame = cv2.imread(
                    os.path.join(args["video"], str(name), str(name) + f"_{num_i}.jpg"))
            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if frame is None:
                break
            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue
            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            # Only take different areas that are different enough (>25 / 255)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # Dilate a bit to make differences more seeable;
            # more suitable for contour detection
            # and to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                    continue
                # save starting frame
                start_new[film_i] = num_i
                succes = False
                break
    df["start"] = start_new
    # save annotation file
    df.to_csv(args["ann_output"], sep=" ", index=False)

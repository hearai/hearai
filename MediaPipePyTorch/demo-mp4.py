import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = True

face_detector = BlazeFace(back_model=back_detector).to(gpu)
if back_detector:
    face_detector.load_weights("blazefaceback.pth")
    face_detector.load_anchors("anchors_face_back.npy")
else:
    face_detector.load_weights("blazeface.pth")
    face_detector.load_anchors("anchors_face.npy")

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")

show=True
WINDOW='test'
threshold=0.3

if len(sys.argv)!=2:
    print("Usage:")
    print("%s file.mp4"%sys.argv[0])
    print("")
    exit(1)


if show:
    cv2.namedWindow(WINDOW)


capture = cv2.VideoCapture(sys.argv[1])

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

print('frame,'+','.join(['x%d,y%d'%(i,i) for i in range(21)])+',confidence')

while hasFrame:
    frame_ct +=1

    frame = np.ascontiguousarray(frame[:,:,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    if back_detector:
        normalized_face_detections = face_detector.predict_on_image(img1)
    else:
        normalized_face_detections = face_detector.predict_on_image(img2)
    normalized_palm_detections = palm_detector.predict_on_image(img1)

    face_detections = denormalize_detections(normalized_face_detections, scale, pad)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)


    xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks = face_regressor(img.to(gpu))
    landmarks = face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)


    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
    img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)
    

    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)


    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]

        print('%d,'%(frame_ct) + ','.join(['%.2f'%(_) for _ in landmark[:,:-1].flatten().tolist()])+',%.2f'%flag)
        sys.stdout.flush()
        if flag>threshold:
            if show:
                draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
    if show:
        draw_roi(frame, box)
        draw_roi(frame, box2)
        draw_detections(frame, face_detections)
        draw_detections(frame, palm_detections)

        cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()

    if show:
        key = cv2.waitKey(1)
        if key == 27:
            break

capture.release()
cv2.destroyAllWindows()

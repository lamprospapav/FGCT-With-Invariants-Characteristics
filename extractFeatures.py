import numpy as np
import cv2

def extractFeatures(image):

    if np.shape(image)[2] == 3 :
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # kp is the keypoints
    # kp.angle = angle of point
    # kp.pt = (x,y) point
    # kp.octave = scale of point
    
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features

    sift =cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(image, None)



    return kp,desc



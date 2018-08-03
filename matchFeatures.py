import cv2


def matchFeatures(logoImages, testImages, dthr):

    # distance - Distance between descriptors. The lower, the better it is
    # trainIdx - Index of the descriptor in train descriptors
    # queryIdx - Index of the descriptor in query descriptors
    # imgIdx   - Index of the train image.

    bf = cv2.BFMatcher (cv2.NORM_L2, crossCheck =True)
    logo_desc = logoImages[3]
    test_desc = testImages[3]

    matches = bf.match(test_desc,logo_desc)
    matches = list(filter(lambda a: a.distance < dthr, matches))
    pairs = sorted(matches, key=lambda x:  x.distance)
    return pairs
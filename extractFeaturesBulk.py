import cv2
import extractFeatures
import glob

def extractFeaturesBulk (path):
    types = ('*.png', '*.jpg')
    listofFiles = []
    for files in types:
        listofFiles.extend(glob.glob(path+'/'+ files))
    elements = [[] for y in listofFiles for i in range(1)]

    for idx, imagepath in enumerate(listofFiles):

        img = cv2.imread(imagepath)
        elements[idx].append(img)
        elements[idx].append(imagepath)
        kp, desc = extractFeatures.extractFeatures(img)
        elements[idx].append(kp)
        elements[idx].append(desc)
    return elements

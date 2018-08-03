import numpy as np
# Create a list.
elements = []

# Append empty lists in first two indexes.
elements.append([])
elements.append([])

# Add elements to empty lists.
elements[0].append(2313)
elements[0].append('fdsafd')
elements[0].append(np.arange(9).reshape(3,3))


elements[1].append(3)
elements[1].append(4)

# Display top-left element.
print(elements[1][1])

# Display entire list.


import os
import fnmatch
import numpy as np
import cv2


def extractFeaturesBulk (path):
    listofFiles = os.listdir(path)
    elements = [[] for y in listofFiles]

    pattern = ".png"
    for idx, val in enumerate(listofFiles):
        imagepath = path + '/' + val
        img = cv2.imread(imagepath,0)
        elements[idx].append(img)
        elements[idx].append(val)
    return elements
plt.imshow(cv2.cvtColor(elements[1][0],cv2.COLOR_BGR2RGB))
elements = [[] for y in listofFiles]

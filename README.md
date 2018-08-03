# FGCT-With-Invariant-Characteristics
This is a different version of **Fast Geometric Consistency Test  [(FGCT)](https://github.com/nzikos/FGCT)** algorithm written in Python. The new method create pairs of triangles that is invariant to projective transformations.
## Overview
FGCT is used for logos/trademarks or object detection and clasification from test images. Method is done in four steps:
- Extract features (SIFT) from test and reference logo image.
- Match test image features with logo images feature in the descriptor space.
- Create triangle from the matching features
- Use triangles and using the new version of FGCT to calculate the corresponding features that forms a consistent geometry on image and logo feature sets.

## Note
IMPORTANT: At first run you need to run
- sudo apt-get install python-pip
- sudo pip install opencv-contrib-python
- sudo pip install matplotlib


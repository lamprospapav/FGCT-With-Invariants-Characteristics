import math
import numpy as np


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def tri_angles(a,b,c):
    side_a = distance(a, b)
    side_b = distance(b, c)
    side_c = distance(c, a)
    angle = np.zeros((3,1),float)
    if (side_a ==0) or (side_b ==0) or (side_c ==0):
        angle[0] = 0
        angle[1] = 0
        angle[2] = 0
    else:
        angle[0] = math.degrees(math.acos((side_a ** 2 + side_b ** 2 - side_c ** 2) / (2.0 * side_a * side_b)))
        angle[1] = math.degrees(math.acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2.0 * side_a * side_c)))
        angle[2] = math.degrees(math.acos((side_c ** 2 + side_b ** 2 - side_a ** 2) / (2.0 * side_c * side_b)))
    return angle


def tri_area(a, b, c):
    side_a = distance(a, b)
    side_b = distance(b, c)
    side_c = distance(c, a)
    s = 0.5 * (side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
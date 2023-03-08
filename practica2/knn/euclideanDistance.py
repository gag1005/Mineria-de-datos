from math import sqrt
import numpy as np

def euclideanDistance(a: tuple, b: tuple) -> float:
    total = 0
    for i in zip(a,b):
        total += (i[1] - i[0]) ** 2
    
    return sqrt(total)

def euclideanDistanceVec(a: np.ndarray, b: np.ndarray):
    return np.sqrt(np.sum(np.square(np.subtract(b, a))))
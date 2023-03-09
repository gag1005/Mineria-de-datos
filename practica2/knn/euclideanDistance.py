from math import sqrt
import numpy as np

def euclideanDistance(a: tuple, b: tuple) -> float:
    total = 0
    for i in zip(a,b):
        total += (i[1] - i[0]) ** 2
    
    return sqrt(total)

def euclideanDistanceVec(a: np.ndarray, b: np.ndarray):
    # return np.sqrt(np.sum(np.square(a - b)))
    # Esto es aquivalente a la línea anterior pero 3 veces mas rápido
    return np.linalg.norm(a - b)
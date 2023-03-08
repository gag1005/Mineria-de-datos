from math import sqrt

def euclideanDistance(a: tuple, b: tuple) -> float:
    total = 0
    for i in zip(a,b):
        total += (i[1] - i[0]) ** 2
    
    return sqrt(total)
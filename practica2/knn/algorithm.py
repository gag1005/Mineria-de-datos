from knn.euclideanDistance import euclideanDistance
from typing import Callable

class Knn:
    trainData: list[tuple]
    trainClass: list
    k: int

    def __init__(self, trainData: list[tuple], trainClass: list, k) -> None:
        self.trainData = trainData
        self.trainClass = trainClass
        self.k = k

    def clasify(self, instance: tuple, distanceFunc: Callable[[tuple[float], tuple[float]], float]=euclideanDistance):
        distances: list[tuple[int, float]] = [(i, 0.0) for i in range(len(self.trainData))]

        for i in range(len(self.trainData)):
            distances[i] = (i, distanceFunc(self.trainData[i], instance))
        
        distances.sort(key=lambda x: x[1])

        classes: dict = {}
        for c in tuple(self.trainClass):
            classes[c] = 0

        elements = distances[:self.k]

        for e in elements:
            classes[self.trainClass[e[0]]] += 1

        higher = (-1, -1)
        c = list(classes.items())
        for i in c:
            if i[1] > higher[1]:
                higher = i

        return higher[0]
from knn.euclideanDistance import euclideanDistance, euclideanDistanceVec
from typing import Callable
import numpy as np
import pandas as pd

class Knn:
    trainData: np.ndarray
    trainClass: np.ndarray
    k: int

    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame) -> None:
        self.trainData = xTrain.to_numpy()
        self.trainClass = yTrain.to_numpy()
        
    def pred(self, xTest: pd.DataFrame):
        test = xTest.to_numpy()
        return np.array([self.predSingle(i) for i in test])

    def predSingle(self, single: np.ndarray):
        size = self.trainData.shape[0]
        distances = np.apply_along_axis(euclideanDistanceVec, 1, self.trainData, single)
        distances = np.stack((np.arange(size), distances)).transpose()
        distances = distances[distances[:, 1].argsort()] # Esto ordena el array según la segunda columna (la distancia)
        

        classes: dict = {}

        for c in np.unique(self.trainClass):
            classes[c] = 0

        elements = distances[:self.k]

        for e in elements:
            classes[self.trainClass[int(e[0])][0]] += 1

        higher = (-1, -1)
        c = list(classes.items())
        for i in c:
            if i[1] > higher[1]:
                higher = i

        return higher[0]
    
    def precision(self, predicted: np.ndarray, real: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
        real = real.to_numpy().transpose()
        # precision = np.sum(np.equal(predicted, real)) / predicted.shape[0]
        numClasses = len(classes)
        confusionMatrix = np.zeros((numClasses, numClasses))

        for i in range(numClasses):
            for j in range(numClasses):
                confusionMatrix[i][j] = np.sum(np.logical_and((predicted == classes[i]), (real == classes[j])))

        
        return confusionMatrix

    

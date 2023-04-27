import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial
import random
import time

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, lRate=0.1, maxIter=1000, maxMSE=0) -> None:
        self.n = xTrain.shape[0]
        # Se le añade una columna de 1 para los atributos independientes
        self.xTrain = np.concatenate((xTrain.to_numpy(), np.ones((self.n, 1))), axis=1)
        self.yTrain = yTrain.to_numpy()
        self.coefs = np.array([[np.float64(random.random()) for x in range(self.xTrain.shape[1])]]).transpose()
        self.maxIter = maxIter
        self.lRate = lRate
        self.mseHistory: list = []
        self.maxMSE = maxMSE

    def train(self) -> None:
        it = 0
        mse = 999999999999999
        while(mse > self.maxMSE and it < self.maxIter):
            vMSE = self.dmse()
            self.coefs -= self.lRate * vMSE
            self.mseHistory.append(mse := self.mse())
            it += 1
        print("Número de iteraciones: " + str(it))
        print("Coeficientes finales: " + str(self.coefs.transpose()[0]))
        print("MSE final: " + str(self.mse()))

    def dmse(self) -> np.ndarray:
        return ((2/self.n) * self.xTrain.transpose()) @ (self.predictions(self.xTrain) - self.yTrain)
         
    def predictions(self, points: np.ndarray) -> np.ndarray:
        return points @ self.coefs

    def mse(self) -> np.ndarray:
        return (1/self.n) * np.sum(np.square(self.predictions(self.xTrain) - self.yTrain))


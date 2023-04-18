import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial
import random
import time

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, numIter, lRate) -> None:
        self.n = xTrain.shape[0]
        # Se le añade una columna de 1 para los atributos independientes
        self.xTrain = np.concatenate((xTrain.to_numpy(), np.ones((self.n, 1))), axis=1)
        self.yTrain = yTrain.to_numpy()
        self.coefs = np.array([[np.float64(random.random()) for x in range(self.xTrain.shape[1])]]).transpose()
        self.numIter = numIter
        self.lRate = lRate
        self.mseHistory: np.ndarray = np.zeros(self.numIter)

    def train(self, calculateMSE=False) -> None:
        it = 0
        for i in range(self.numIter):
            vMSE = self.dmse()
            # print(vMSE.transpose()[0])
            if calculateMSE:
                self.mseHistory[i] = self.mse()
            self.coefs -= self.lRate * vMSE
            it += 1
        print("Coeficientes finales: " + str(self.coefs.transpose()[0]))
        print("MSE final: " + str(self.mse()))

    def dmse(self) -> np.ndarray:
        pred = self.predictions()
        vMSE = ((2/self.n) * self.xTrain.transpose()) @ (pred - self.yTrain)
        return vMSE
    
    def predictions(self) -> np.ndarray:
        return self.xTrain @ self.coefs

    def mse(self) -> np.ndarray:
        pred = self.predictions()
        mse = (1/self.n) * np.sum(np.square(pred - self.yTrain))
        return mse
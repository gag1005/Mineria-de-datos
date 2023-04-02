import numpy as np
import pandas as pd

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, numIter, lRate) -> None:
        self.xTrain = xTrain.to_numpy()
        self.yTrain = yTrain.to_numpy()
        self.coefs = np.zeros(xTrain.shape[1] + 1)
        self.numIter = numIter
        self.lRate = lRate
        self.n = xTrain.shape[0]

    def train(self):
        for i in range(5):
            coefVars = np.array([self.dmse(x) for x in range(self.coefs.size)])
            print(coefVars)
            self.coefs = self.coefs - self.lRate * coefVars
            print(self.coefs)

    def dmse(self, coefnum:int) -> float:
        difs = np.apply_along_axis(lambda x: np.append(x[:-1], 1)[coefnum] * (self.predict(x[:-1]) - x[-1]), 1, np.concatenate((self.xTrain, self.yTrain), axis=1))
        coefVar = -2 * np.sum(difs) / self.n
        return coefVar
    
    def predict(self, point:np.ndarray) -> float:
        return np.sum(self.coefs[:-1] * point) + self.coefs[-1]
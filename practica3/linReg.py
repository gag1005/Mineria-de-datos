import numpy as np
import pandas as pd

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, numIter:int=1000, lRate:float=0.001) -> None:
        self.xTrain = xTrain.to_numpy()
        self.yTrain = yTrain.to_numpy()
        #self.coefs = np.random.rand(xTrain.shape[1] + 1)
        self.coefs = np.zeros(xTrain.shape[1] + 1)
        self.numIter = numIter
        self.lRate = lRate
        self.n = xTrain.shape[0]

    def train(self):
        for i in range(10):
            print(np.stack((self.coefs, self.dmse())))
            self.coefs = np.apply_along_axis(lambda x: x[0] - self.lRate * x[1], 0, np.stack((self.coefs, self.dmse())))
        
        print(self.coefs)


    # def mse(self) -> float:
    #     # TODO: Probar con np.square a ver si es mas rápido
    #     return (1 / self.xTrain.shape[0]) * np.sum(np.apply_along_axis(lambda x: (self.getValue(x[:-1]) - x[-1]) ** 2, 1, np.concatenate((self.xTrain, self.yTrain), axis=1)))
    
    def dmse(self) -> np.ndarray:
        difs = (self.yTrain.transpose() - np.apply_along_axis(self.getValue, 1, self.xTrain))[0]
        coefVar = np.apply_along_axis(lambda x: ((-2) / self.n) * np.sum(difs), 0, np.array([self.coefs]))
        return coefVar
    
    def getValue(self, point:np.ndarray) -> float:
        return np.sum(self.coefs[:-1] * point) + self.coefs[-1]
        
    


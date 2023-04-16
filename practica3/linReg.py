import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial

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
            # coefVars = np.array([self.dmse(x) for x in range(self.coefs.size)])
            nums = np.array([np.arange(self.coefs.size)])
            coefVars = np.apply_along_axis(self.dmse, 0, nums)
            print(coefVars)
            print(self.coefs)
            self.coefs -= self.lRate * coefVars
            print(self.coefs)
            print("")

    def dmse(self, coefnum:int) -> float:
        difs = self.difs(coefnum)
        coefVar = -2 * np.sum(difs) / self.n
        return coefVar
    
    def predict(self, point:np.ndarray) -> float:
        return np.sum(self.coefs[:-1] * point) - self.coefs[-1]
    
    def difs(self, coefnum):
        pool = Pool(processes=multiprocessing.cpu_count())
        csize, extra = divmod(self.n, multiprocessing.cpu_count())
        csize += not not extra
        difs = np.array(list(pool.imap(partial(self.func, coefnum), np.concatenate((self.xTrain, self.yTrain), axis=1), chunksize=csize)))
        pool.close()
        return difs

    def func(self, coefnum, x):
        return np.append(x[:-1], 1)[coefnum] * (abs(self.predict(x[:-1]) - x[-1]))
    
    def predict2d(self, value, coefnum):
        return self.coefs[coefnum] * value + self.coefs[-1]
    
        # difs = np.apply_along_axis(lambda x: np.append(x[:-1], 1)[coefnum] * (self.predict(x[:-1]) - x[-1]), 1, np.concatenate((self.xTrain, self.yTrain), axis=1))

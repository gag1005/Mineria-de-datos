import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial
import random

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, numIter, lRate) -> None:
        self.xTrain = xTrain.to_numpy()
        self.yTrain = yTrain.to_numpy()
        # print(self.xTrain)
        # print([random.randrange(np.min(self.xTrain[x]), np.max(self.xTrain[x])) for x in range(xTrain.shape[1] + 1)])
        # self.coefs = np.array([np.float64(random.randrange(int(np.min(self.xTrain[x])), int(np.max(self.xTrain[x])))) for x in range(self.xTrain.shape[1] + 1)])
        # self.coefs = np.array([np.float64(random.randrange(int(np.min(self.xTrain[x])), int(np.max(self.xTrain[x]))))) for x in range(self.xTrain.shape[1] + 1)])
        
        self.coefs = np.zeros(self.xTrain.shape[1] + 1) + 1
        
        self.numIter = numIter
        self.lRate = lRate
        self.n = xTrain.shape[0]
        self.d:list= []

    def train(self):
        for i in range(self.numIter):
            # coefVars = np.array([self.dmse(x) for x in range(self.coefs.size)])
            nums = np.array([np.arange(self.coefs.size)])
            coefVars = np.apply_along_axis(self.dmse, 0, nums)
            # msevars = np.apply_along_axis(self.mse, 0, nums)
            print(coefVars)
            # print(msevars)
            self.coefs -= self.lRate * coefVars
            print(self.coefs)
            print("")

    def dmse(self, coefnum:int) -> float:
        difs = self.difs(coefnum)
        coefVar = np.sum(difs) / self.n
        # if coefnum == 1:
            # self.d.append(coefVar)
        return coefVar

    
    def difs(self, coefnum: int):
        # pool = Pool(processes=multiprocessing.cpu_count())
        # csize, extra = divmod(self.n, multiprocessing.cpu_count())
        # csize += not not extra
        # difs = np.array(list(pool.imap(partial(self.dmseFunc, coefnum), np.concatenate((self.xTrain, self.yTrain), axis=1), chunksize=csize)))
        # pool.close()
        # print(np.concatenate((self.xTrain, self.yTrain), axis=1))
        difs = np.apply_along_axis(lambda x: self.dmseFunc(coefnum, x), 1, np.concatenate((self.xTrain, self.yTrain), axis=1))

        return difs
    
    def predict(self, point:np.ndarray) -> float:
        return np.sum(self.coefs[:-1] * point) + self.coefs[-1]

    def dmseFunc(self, coefnum, x):
        # print("-> " + str(np.append(x[:-1], 1)[coefnum]))
        return np.append(x[:-1], 1)[coefnum] * (self.predict(x[:-1]) - x[-1])
    














    
    def predict2d(self, value, coefnum):
        return self.coefs[coefnum] * value + self.coefs[-1]
    
        # difs = np.apply_along_axis(lambda x: np.append(x[:-1], 1)[coefnum] * (self.predict(x[:-1]) - x[-1]), 1, np.concatenate((self.xTrain, self.yTrain), axis=1))
    




    def mse(self, coefnum:int) -> float:
        difs = self.difsmse(coefnum)
        coefVar = np.sum(difs) / self.n
        # if coefnum == 1:
        #     self.d.append(coefVar)
        return coefVar

    def difsmse(self, coefnum: int):
        pool = Pool(processes=multiprocessing.cpu_count())
        csize, extra = divmod(self.n, multiprocessing.cpu_count())
        csize += not not extra
        difs = np.array(list(pool.imap(partial(self.mseFunc, coefnum), np.concatenate((self.xTrain, self.yTrain), axis=1), chunksize=csize)))
        pool.close()
        return difs

    def mseFunc(self, coefnum, x):
        return np.append(x[:-1], 1)[coefnum] * ((abs(self.predict(x[:-1]) - x[-1])) ** 2)
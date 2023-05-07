import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from functools import partial
import random
import time

class LinRegClassifier:

    def __init__(self, xTrain:pd.DataFrame, yTrain:pd.DataFrame, numIter, lRate) -> None:
        self.xTrain = xTrain.to_numpy()
        self.yTrain = yTrain.to_numpy()
        # print(self.xTrain)
        # print((np.min(self.xTrain[0]), np.max(self.xTrain[0])))
        # print([random.randrange(0, 1) for x in range(self.xTrain.shape[1] + 1)])
        self.coefs = np.array([np.float64(random.randrange(0, 1)) for x in range(self.xTrain.shape[1] + 1)])
        # self.coefs = np.array([np.float64(random.randrange(int(np.min(self.xTrain[x])), int(np.max(self.xTrain[x]))))) for x in range(self.xTrain.shape[1] + 1)])
        
        # self.coefs = np.zeros(self.xTrain.shape[1] + 1) + 1
        
        self.numIter = numIter
        self.lRate = lRate
        self.n = xTrain.shape[0]
        self.mseHistory: np.ndarray = np.zeros(self.numIter)

        self.csize = 1
        self.csize, self.extra = divmod(self.n, multiprocessing.cpu_count())
        self.csize += not not self.extra
        

        self.csize



    def train(self, calculateMSE=False):
        it = 0
        maxMSE = 1
        
        pool = Pool(processes=multiprocessing.cpu_count())
        nums = np.array([np.arange(self.coefs.size)])
        for i in range(self.numIter):
            coefVars = np.apply_along_axis(self.dmse, 0, nums, pool)
            self.coefs -= self.lRate * coefVars
            if calculateMSE:
                msevars = np.apply_along_axis(self.mse, 0, nums)
                maxMSE = np.max(msevars)
                self.mseHistory[i] = maxMSE
            print("Iter: " + str(it) + (" max MSE: " + str(maxMSE) if calculateMSE else ""))
            it += 1
        pool.close()


    def dmse(self, coefnum:int, pool) -> float:
        start = time.time()
        print("1: " + str(time.time() - start))
        difs = self.difs(coefnum, pool)
        print("2: " + str(time.time() - start))
        coefVar = np.divide(np.sum(difs), self.n)
        print("3: " + str(time.time() - start))
        return coefVar

    
    def difs(self, coefnum: int, pool):
        difs = np.array(list(pool.map(partial(self.dmseFunc, coefnum), np.concatenate((self.xTrain, self.yTrain), axis=1), chunksize=self.csize))).transpose()
        # difs = np.apply_along_axis(lambda x: self.dmseFunc(coefnum, x), 1, np.concatenate((self.xTrain, self.yTrain), axis=1))
        return difs
    
    def predict(self, point:np.ndarray) -> float:
        return np.sum(self.coefs[:-1] * point) + self.coefs[-1]

    def dmseFunc(self, coefnum, x):
        return np.append(x[:-1], 1)[coefnum] * (self.predict(x[:-1]) - x[-1])
    
    def mse(self, coefnum:int) -> float:
        difs = self.difsmse(coefnum)
        coefVar = np.sum(difs) / self.n
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
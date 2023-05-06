import pandas as pd
from sklearn.base import ClassifierMixin
import numpy as np

class ClassifierEnsemble(ClassifierMixin):
    def __init__(self, classifierList: list[tuple[str, ClassifierMixin]], weights: list[float] = []) -> None:
        self.classifierList = classifierList
        self.weights = [1 for i in range(len(classifierList))] if weights == [] else self.weights
        self.splits: list[pd.DataFrame]
        self.n: int

    def fit(self, xx: pd.DataFrame, yy: pd.DataFrame):
        x = xx.to_numpy()
        y = yy.to_numpy()
        self.n = yy.size
        self.splitData(xx, yy)

        # for c, d in zip(self.classifierList, self.splits): c[1].fit(d)

    def predict(self):
        pass

    def splitData(self, x: pd.DataFrame, y: pd.DataFrame):
        self.splits = [pd.concat((x, y)).sample(int(self.n / len(self.classifierList))) for i in range(len(self.classifierList))]

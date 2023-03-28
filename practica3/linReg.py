import numpy as np

def linReg(data:np.ndarray) -> np.ndarray:
    coefs = np.random.rand(data.size + 1)
    print(mse(data, coefs))
    return np.array([])

def getLineValue(coefs: np.ndarray, variables: np.ndarray) -> np.ndarray:
    return coefs[:-1] * variables + coefs[-1]


def mse(a:np.ndarray, b:np.ndarray) -> float:
    return np.sum(np.square(a - b))


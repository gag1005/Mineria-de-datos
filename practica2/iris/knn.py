# from myknn import euclideanDistanceVec
import numpy as np
import pandas as pd

def knn_distancias(xTrain,xTest,k):
    """
    Encuentra los k vecinos más cercanos de xTest en xTrain.
    Entrada:
        xTrain = matriz n x d. n=filas y d=características
        xTest = matriz m x d. m=rows y d=features (la misma cantidad de características que xTrain)
        k = número de vecinos más cercanos que se pueden encontrar
    Salida:
        dists = distancias entre todos los puntos xTrain y XTest. Tamaño de n x m
        índices = matriz k x m con los índices de las etiquetas yTrain que representan el punto

    """
    #the following formula calculates the Euclidean distances.

    #returning the top-k closest distances.

    result = np.apply_along_axis(knn_distancias_single, 1, xTest, xTrain, k)

    return result[:,:,1], result[:,:,0]

def knn_distancias_single(single, xTrain, k):
    size = xTrain.shape[0]
    distances = np.apply_along_axis(euclideanDistanceVec, 1, xTrain, single)
    distances = np.stack((np.arange(size), distances)).transpose()
    distances = distances[distances[:, 1].argsort()][:k] # Esto ordena el array según la segunda columna (la distancia)
    return distances


def knn_predecir(xTrain,yTrain,xTest,k=3):
    """
    Utiliza xTrain e yTrain para predecir xTest.
    Entrada:
    xTrain = matriz n x d. n=filas y d=características
    yTrain = n x 1 matriz. n=filas con valor de etiqueta
    xTest = matriz m x d. m=rows y d=features (la misma cantidad de características que xTrain)
    k = número de vecinos más cercanos que se pueden encontrar
    Salida:
    predicciones = etiquetas predichas, es decir, preds(i) es la etiqueta predicha de xTest(i,:)

    """
    yTrain = yTrain.to_numpy()
    dists, inds = knn_distancias(xTrain, xTest, k)
    getTag = np.vectorize(lambda x: yTrain[int(x)][0])
    tags = getTag(inds)
    predicts = np.apply_along_axis(lambda x: pd.DataFrame(x).mode()[0][0], 1, tags)
    # predicts = np.array([pd.DataFrame(x).mode()[0].to_numpy()[0] for x in tags])
    return predicts.transpose()

def knn_precision(yTest, predictions):
    """
    Evalúa la precisión del knn_predictions. Devuelve un valor entre 0 y 100%.
    Entrada:
    yTest = matriz de forma (n,) donde n = filas en el conjunto de prueba
    preds = matriz de forma (n,) donde n = filas en el conjunto de prueba
    Salida:
    precisión = % de respuestas correctas en la predicción
    """

    return np.sum(predictions == yTest.to_numpy().transpose()) / predictions.shape[0]

def euclideanDistanceVec(a: np.ndarray, b: np.ndarray):
    # return np.sqrt(np.sum(np.square(a - b)))
    # Esto es aquivalente a la línea anterior pero 3 veces mas rápido
    return np.linalg.norm(a - b)


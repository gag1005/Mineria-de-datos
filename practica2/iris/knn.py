import numpy as np
import pandas as pd
from collections import Counter

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

    # Se aplica el método knn_distancias_single a cada elemento de xTest
    result = np.apply_along_axis(knn_distancias_single, 1, xTest, xTrain, k)
    return result[:,:,1], result[:,:,0]

def knn_distancias_single(single, xTrain, k):
    # Se obtiene el número de elementos
    size = xTrain.shape[0]
    # Se calculan las distancias a cada punto
    distances = np.apply_along_axis(euclideanDistanceVec, 1, xTrain, single)
    # Se une el array de distancias a un array de índices
    distances = np.stack((np.arange(size), distances)).transpose()
    # Se ordena la matriz según la columna de distancias y se escojen los k primeros
    distances = distances[distances[:, 1].argsort()][:k]
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
    # Se convierte el DataFrame yTrain a un array de numpy
    yTrain = yTrain.to_numpy()
    # Se calculan las distancias e índices de los puntos mas cercanos a cada punto de xTest
    # las distancias son innecesarias
    dists, inds = knn_distancias(xTrain, xTest, k)
    # Se convierte una función de un único elemento en vectorial (funciona con varios elementos)
    getTag = np.vectorize(lambda x: yTrain[int(x)][0])
    # Se aplica la función anterior al array inds para obtener las etiquetas que corresponden a cada índice
    tags = getTag(inds)
    # Se obtiene por cada elemento la etiqueta mas común
    predicts = np.apply_along_axis(lambda x: Counter(x).most_common()[0], 1, tags)
    return predicts[:, 0].transpose()

    # return np.apply_along_axis(lambda x: Counter(x).most_common()[0], 1, np.vectorize(lambda x: yTrain.to_numpy()[int(x)][0])(knn_distancias(xTrain, xTest, k)[1]))[:, 0].transpose()

def knn_precision(yTest, predictions):
    """
    Evalúa la precisión del knn_predictions. Devuelve un valor entre 0 y 100%.
    Entrada:
    yTest = matriz de forma (n,) donde n = filas en el conjunto de prueba
    preds = matriz de forma (n,) donde n = filas en el conjunto de prueba
    Salida:
    precisión = % de respuestas correctas en la predicción
    """

    return (np.sum(predictions == yTest.to_numpy().transpose()) / predictions.shape[0]) * 100

def euclideanDistanceVec(a: np.ndarray, b: np.ndarray):
    # return np.sqrt(np.sum(np.square(a - b)))
    # Esto es aquivalente a la línea anterior pero 3 veces mas rápido
    return np.linalg.norm(a - b)

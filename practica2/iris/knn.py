# from myknn import euclideanDistanceVec
import numpy as np

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
    dists = np.array()
    inds = np.array()
    for ind in xTest:
        d, i = knn_distancias_single(xTrain, ind, k)

        # distst = np.concatenate(, axis=1)
    results = [knn_distancias_single(xTrain, ind, k) for ind in xTest]
    return np.concatenate(results, axis=1)

    

    

def knn_distancias_single(xTrain, single, k):
    size = xTrain.shape[0]
    distances = np.apply_along_axis(euclideanDistanceVec, 1, xTrain, single)
    distances = np.stack((np.arange(size), distances)).transpose()
    distances = distances[distances[:, 1].argsort()] # Esto ordena el array según la segunda columna (la distancia)
    return distances[1], distances[0]


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

    predicciones = np.array([knn_pred_single(xTrain, yTrain, xTest, k, distances) for distances in knn_distancias(xTrain, xTest, k)])
    
  
    return predicciones

def knn_pred_single(xTrain, yTrain, xTest, k, distances):
    classes: dict = {}

    for c in np.unique(yTrain):
        classes[c] = 0

    elements = distances[:k]

    for e in elements:
        classes[yTrain[e[0]][0]] += 1

    higher = (-1, -1)
    c = list(classes.items())
    for i in c:
        if i[1] > higher[1]:
            higher = i

    return higher[0]

def knn_precision(yTest,predictions):
    """
    Evalúa la precisión del knn_predictions. Devuelve un valor entre 0 y 100%.
    Entrada:
    yTest = matriz de forma (n,) donde n = filas en el conjunto de prueba
    preds = matriz de forma (n,) donde n = filas en el conjunto de prueba
    Salida:
    precisión = % de respuestas correctas en la predicción
    """


def euclideanDistanceVec(a: np.ndarray, b: np.ndarray):
    # return np.sqrt(np.sum(np.square(a - b)))
    # Esto es aquivalente a la línea anterior pero 3 veces mas rápido
    return np.linalg.norm(a - b)


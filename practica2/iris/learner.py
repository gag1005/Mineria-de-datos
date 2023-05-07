import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.DataFrame(input_table)

x = df.iloc[:, 1:]
y = df.iloc[:, 0]

knn = KNeighborsClassifier(n_neighbors=flow_variables['integer-input'])

output_model = knn.fit(x, y)
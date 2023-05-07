import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.DataFrame(input_table)

x = df.iloc[:, 1:]
y = pd.DataFrame(df.iloc[:, 0])

print(x)

prediction = input_model.predict(x)

output_table = pd.DataFrame()
output_table['yTest'] = y
output_table['prediction'] = prediction
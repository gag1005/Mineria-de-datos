import math
import numpy as np
import pandas as pd
from datetime import datetime

def coordsToDistance(long1: float, lat1: float, long2: float, lat2: float):
    # Radio de la tierra aprox
    R = 6367000
    # Grados a radianes
    degToRad = math.pi / 180

    distLong = (long2 - long1) * degToRad
    distLat = (lat1 - lat2) * degToRad
    
    a = (math.sin(distLat / 2) ** 2) + (math.cos(lat1 * degToRad) * math.cos(lat2 * degToRad) * (math.sin(distLong / 2) ** 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def transformDataframeCoordsIntoDistance(df: pd.DataFrame, cols: list[str], newColName: str):
    # ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    coords:np.ndarray = df[cols].to_numpy()
    distances = np.apply_along_axis(lambda x: coordsToDistance(x[0], x[1], x[2], x[3]), 1, coords)
    df.insert(2, newColName, pd.Series(distances))

def transformDates(df: pd.DataFrame):
    df['pickup_timestamp'] = df['pickup_datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
    df['pickup_hour'] = df['pickup_timestamp'].apply(lambda x: x.hour)
    df.drop('pickup_datetime', axis=1, inplace=True)
    df.drop('pickup_timestamp', axis=1, inplace=True)



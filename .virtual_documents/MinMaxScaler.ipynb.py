import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, minmax_scale


df = pd.read_csv("./Datasets/absenteeism_processed.csv")

df.head()


scaled_df = pd.DataFrame(minmax_scale(df, feature_range=(0, 5)), columns = df.columns)
scaled_df.boxplot(figsize = (10, 10))

plt.show();


height_df = pd.DataFrame(df.Height, columns = ["Height"])

height_df.head()


height_df["scaled"] = scale(height_df)

height_df.head()


# num op that min max scaler does under hood
range_max = 100
range_min = 0

height_max = height_df["Height"].max()  # gets max height
height_min = height_df["Height"].min()  # gets min height


height_df["range_scaled"] = ((height_df.Height - height_min) / (height_max - height_min)) * \
                            (range_max - range_min) + range_min

height_df.head()


# compare result from manual minmax with sklearns own
height_df["minmax_scaled"] = minmax_scale(height_df["Height"], feature_range=(0, 100))

height_df.head()


# minmax scale using estimator
minmax_scaler = MinMaxScaler(feature_range=(0, 100))

height_df["minmax_estimator_scaled"] = minmax_scaler.fit_transform(height_df.Height.values.reshape(-1, 1))

height_df.head()





























































































































































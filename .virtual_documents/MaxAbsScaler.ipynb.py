import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MaxAbsScaler, maxabs_scale


df = pd.read_csv("./Datasets/absenteeism_processed.csv")

df.head()


scaled_df = pd.DataFrame(maxabs_scale(df), columns = df.columns)

scaled_df.boxplot(figsize = (10, 10))
plt.show();


# gets min and max value of scaled dataset
scaled_df.min(), scaled_df.max()


# MaxAbs Scaling using the estimator obj
maxabs_scaler = MaxAbsScaler()

maxabs_scaled_array = maxabs_scaler.fit_transform(df)
maxabs_scaled_df = pd.DataFrame(maxabs_scaled_array, columns = df.columns)

maxabs_scaled_df.boxplot(figsize = (10, 10))
plt.show();
















































































































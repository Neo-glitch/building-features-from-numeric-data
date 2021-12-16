import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer, normalize, StandardScaler, scale


driver_df = pd.read_csv("./Datasets/data_1024.csv", delimiter="\t")

driver_df.head()


# get just 2 needed features
driver_df = driver_df[["Distance_Feature", "Speeding_Feature"]]

driver_df = driver_df.astype(np.float32)

driver_df.dtypes


# data viz
fig, ax = plt.subplots(figsize = (10,10))
plt.scatter(driver_df["Distance_Feature"], driver_df["Speeding_Feature"])


normalized_l2_df = pd.DataFrame(normalize(driver_df, norm="l2"), columns = driver_df.columns)
normalized_l2_df.head()


# check if actual norm of each row is == 1(unit norm) and store in df
normalized_l2_df["L2"] = normalized_l2_df["Distance_Feature"] * normalized_l2_df["Distance_Feature"] + \
        normalized_l2_df["Speeding_Feature"] * normalized_l2_df["Speeding_Feature"]

normalized_l2_df.head()


normalized_l1_df = pd.DataFrame(normalize(driver_df, norm="l1"), columns = driver_df.columns)
normalized_l1_df.head()


# data viz
fig, ax = plt.subplots(figsize = (10,10))
plt.scatter(normalized_l1_df["Distance_Feature"], normalized_l1_df["Speeding_Feature"])


# to confirm vectors have been scaled to have her values sum up to 1
normalized_l1_df["L1"] = normalized_l1_df["Distance_Feature"] + normalized_l1_df["Speeding_Feature"]

normalized_l1_df.head()


normalized_max_df = pd.DataFrame(normalize(driver_df, norm = "max"), columns = driver_df.columns)

normalized_max_df.head()



















































































































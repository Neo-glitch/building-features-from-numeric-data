import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import cosine_similarity  # cos sim used to check sim btw feature vectors(2 rows or records)
from sklearn.preprocessing import Normalizer, normalize


driver_df = pd.read_csv("./Datasets/data_1024.csv", delimiter="\t")

driver_df.head()


# gets just distance and speeding features
driver_df = driver_df[["Distance_Feature", "Speeding_Feature"]]

driver_df = driver_df.astype("float64")

driver_df.describe()


fig, ax = plt.subplots(figsize = (10,10))
plt.scatter(driver_df["Distance_Feature"], driver_df["Speeding_Feature"])


# helper fun that measures cosine sim, manually
def cosine_similarity_manual(d1, d2):
    x1 = d1[0]  # distance feature of vector(row) 1
    y1 = d1[1]  # speed feature of vector(row) 2
    
    x2 = d2[0]
    y2 = d2[1]
    
    magnitude = math.sqrt(x1*x1 + y1*y1) * math.sqrt(x2*x2 + y2*y2)
    
    dot_product = x1*x2 + y1*y2
    return dot_product / magnitude


# selects 3 drivers at index 0, 1 and 8 from dataset
d1 = driver_df.loc[0]
d2 = driver_df.loc[1]
d3 = driver_df.loc[8]


# checks cos sim btw d1 and d2
cosine_similarity_manual(d1, d2)


cosine_similarity_manual(d1, d3)


# does cos sim using one from sklearn
cosine_similarity(d1.values.reshape(1,-1), d2.values.reshape(1, -1))


# new df with data normalized now
normalized_df = pd.DataFrame(normalize(driver_df, norm="l2"), columns = driver_df.columns)


normalized_df.head()


# viz data
fig, ax = plt.subplots(figsize = (10,10))
plt.scatter(normalized_df["Distance_Feature"], normalized_df["Speeding_Feature"])


# check if actual norm of each row is == 1(unit norm) and store in df
normalized_df["magnitude"] = normalized_df["Distance_Feature"] * normalized_df["Distance_Feature"] + \
        normalized_df["Speeding_Feature"] * normalized_df["Speeding_Feature"]


normalized_df.head()


# helper fun to get cosine sim of normalized vector
def cosine_similarity_manual_normalized(d1, d2):
    x1 = d1[0]  # distance feature of vector(row) 1
    y1 = d1[1]  # speed feature of vector(row) 2
    
    x2 = d2[0]
    y2 = d2[1]
    
    magnitude = 1    # since mag of all norm vector is 1
    
    dot_product = x1*x2 + y1*y2
    return dot_product / magnitude


# selects 3 drivers at index 0, 1 and 8 from dataset
d1 = normalized_df.loc[0]
d2 = normalized_df.loc[1]
d3 = normalized_df.loc[8]


cosine_similarity_manual_normalized(d1, d2)


# clustering, while choosing similarity meaure should be cos sim
import tensorflow as tf

# kmeans algo instance
kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=3, distance_metric="cosine")


# helper fn(input_fn) to convert our input data to format(tensor) needed by tf
def input_fn(x):
    return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(x, dtype = tf.float32),
                                num_epochs = 1)

drivers_array = driver_df.values.astype("float64")
drivers_array


# runs the kmeans clustering for 5 times
num_iterations = 5
for _ in range(num_iterations):
    kmeans.train(input_fn = lambda: input_fn(drivers_array), steps = 1)
    cluster_centers = kmeans.cluster_centers()
    
print("\n\nCluster Centers:\n", cluster_centers)







































































































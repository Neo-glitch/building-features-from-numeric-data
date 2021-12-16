import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer, binarize


# trying bin lib on a dummy dataset
num_list = [[-1000, 0],
           [500, -3000],
           [100, 650]]


binarizer = Binarizer()
binarizer


binarizer.fit(num_list)   # does nothing since by defalt threshold is already specified(i.e 0)


binarized_list = binarizer.transform(X = num_list)

binarized_list


# bin with user defined threshold
binarizer = Binarizer(threshold=500)
binarizer.fit_transform(num_list)


# bin with thresh of 0 for col 0 and 100 for col 1
binarizer = Binarizer(threshold=(0, 100))
binarizer.fit_transform(num_list)


diet_data = pd.read_csv("./Datasets/diet_data.csv")

diet_data.head()


# data cleanup
diet_data.dropna(inplace = True)

diet_data = diet_data.drop(["Date", "Stone", "Pounds", "Ounces"], axis =1)


diet_data = diet_data.astype("float64")

diet_data.head()


# get the median cal content to know when user over ate or when he went a lil hungry
median_calories = diet_data["calories"].median()

median_calories


# binarizer with thresh set at median calories
binarizer = Binarizer(threshold=median_calories)

diet_data["calories_above_median"] = binarizer.fit_transform(diet_data[["calories"]])

diet_data.head()









































































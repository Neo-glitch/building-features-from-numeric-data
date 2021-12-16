import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# helper fn to gen poly data
def true_polynomial_fn(x):
    return 5*x*x*x + 3*x*x + 2*x + 12


np.random.seed(0)

n_samples = 30
degree = 3


X = np.sort(np.random.rand(n_samples))   # gen 30 diff X values
x = X.reshape(-1, 1)                     # reshape to 2d

y = true_polynomial_fn(x) + np.random.rand(n_samples) * 0.1


# viz data to check if polynomial
plt.figure(figsize = (10, 10))
plt.plot(x, y)

plt.show();


# try fitting lin reg model on data 
linear_regression_underfitted = LinearRegression().fit(x, y)

y_pred_underfitted = linear_regression_underfitted.predict(x)


fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(x, y, color = "green", label= "y")
plt.plot(x, y_pred_underfitted, color = "red", label = "y_pred_underfitted")

plt.title("Y")
plt.legend();


# poly features
polynomial_features = PolynomialFeatures(degree=3, include_bias=True)

polynomial_features.fit_transform(x)    # gen poly comb of x features for bias(since included), x1, x2 and x3


linear_regression = LinearRegression()
pipeline = Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", linear_regression)
])


pipeline.fit(x, y)
y_predicted = pipeline.predict(x)


plt.figure(figsize=(10, 10))

plt.plot(x, y, color = "green", label= "y")
plt.plot(x, y_predicted, color = "blue", label = "y_predicted")

plt.title("Y")




































































































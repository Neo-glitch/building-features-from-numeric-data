import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression


analyst_data = pd.read_csv("./Datasets/Analyst_Forecasts.csv")
analyst_data.head()

# values for each row from col q1 - q8 shows how much analyst predictions for a qaurter deviated from actual result
# votes is popularity of analyst


X = analyst_data[["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]]

y = analyst_data[["Votes"]]


# viz data
plt.figure(figsize = (10, 10))

plt.hist(analyst_data["Votes"], facecolor = "green", edgecolor = "red", bins = 40)

plt.xlabel("# Votes")
plt.ylabel("Count")

plt.title("Histogram of # Votes")
plt.show();


# gets the var of each analyst forecast and save in the main df
analyst_data["Variability"] = analyst_data[["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]].var(axis = 1)

analyst_data.head()


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(analyst_data["Variability"], facecolor = "purple", edgecolor = "red", bins = 40)

plt.xlabel("# Variability")
plt.ylabel("Count")

plt.title("Histogram of # Variabilty(Chi Squared distro)")
plt.show();


# fit model to get num of votes/popularity of each analyst
linear_regression = LinearRegression().fit(X, y)
linear_regression.score(X, y)


# compare predictions of poor model to actual values
y_predict = linear_regression.predict(X)

results_df = pd.DataFrame({"y_actual": y["Votes"],
                          "y_predicted": y_predict.reshape(1, -1)[0]})

results_df.head()


# store variablity of analyst pred in another df
x_chi_sq = analyst_data[["Variability"]]
x_chi_sq.head()


# power transforms variablity(chi-squared distro) to normal distro
# using box cox, data must be all +ve.. if data has -ve values use ; "yeo-johnson"
power_transformer = PowerTransformer(method = "box-cox")

x_transformed = power_transformer.fit(x_chi_sq).transform(x_chi_sq)


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(x_transformed, facecolor = "cyan", edgecolor = "red", bins = 40)

plt.xlabel("# Variability")
plt.ylabel("Count")

plt.title("Histogram of # Variabilty(transformed to normal distro)")
plt.show();


# used the transormed variability to predict vote
linear_regression = LinearRegression().fit(x_transformed, y)

linear_regression.score(x_transformed, y)











































































































































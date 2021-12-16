import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes = True)
from sklearn.preprocessing import QuantileTransformer


store_visits = pd.read_csv("./Datasets/store_visits.csv")

store_visits.sample(10)


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(store_visits["Visits"], facecolor = "lime", edgecolor = "red", bins = 40)

plt.xlabel("Visits")
plt.ylabel("Count")

plt.show();

# this shows 2 sep normal distro


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(store_visits["Revenue"], facecolor = "blue", edgecolor = "red", bins = 40)

plt.xlabel("Revenue")
plt.ylabel("Count")
plt.title("Hist of Revenue")

plt.show();

# this shows 2 sep normal distro, hence data is not a single normal distro


# use Quantile transformer to make it single normal distro
transformer = QuantileTransformer(output_distribution="normal")

store_transform = transformer.fit_transform(store_visits[["Visits", "Revenue"]])


# create new df with the tranformed data
store_transform_df = pd.DataFrame(data = store_transform, columns = ["Visits_transform", "Revenue_transform"])

store_transform_df.head()


# concat tranformed df with original df
final_df = pd.concat([store_visits, store_transform_df], axis = 1)
final_df.head()


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(final_df["Visits_transform"], facecolor = "lime", edgecolor = "red", bins = 40)

plt.xlabel("Tranformed_Visits")
plt.ylabel("Count")
plt.title("Histogram of transformed visits")

plt.show();

# shows one distro


# viz data(distor is chi sqaured)
plt.figure(figsize = (10, 10))

plt.hist(final_df["Revenue_transform"], facecolor = "lime", edgecolor = "red", bins = 40)

plt.xlabel("Tranformed_Revenue")
plt.ylabel("Count")
plt.title("Histogram of transformed revenue")

plt.show();

# shows one normal distro


# use sns to plot original data and the regression line of best fit
sns.lmplot(x = "Visits", y = "Revenue", data = final_df, height = 8)
plt.show();


# use sns to plot transformed data and the regression line of best fit
sns.lmplot(x = "Visits_transform", y = "Revenue_transform", data = final_df, height = 8)
plt.show();

















































































































































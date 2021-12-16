import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


# outlier can be special cases where students studies longer and get lower score
# or student doesn't study n get higher scores
student_marks = pd.read_csv("./Datasets/student_performance.csv")
student_marks.sample(10)


fig, ax = plt.subplots(figsize = (10, 10))

plt.scatter(student_marks["Hours Studied"], student_marks["Score Obtained"], color ="red")

ax.set(xlabel= "Hours Studied", ylabel= "Score Obtained", title = "Students Performance")
ax.grid()

plt.show()


# kbin discretizer, unioform startegy means all bins have equal widths
k_bins = KBinsDiscretizer(n_bins = 3, encode = "ordinal", strategy="uniform")


k_bins_array = k_bins.fit_transform(student_marks[["Score Obtained", "Hours Studied"]])
k_bins_array.shape   # values in the 2 cols are bins to which each record in feature belong


kbins_df = pd.DataFrame(data= k_bins_array, columns = ["Binned_Score", "Binned_Hours"])

kbins_df.sample(10)


# concat bin df and original df
students_kbins = pd.concat([student_marks, kbins_df], axis = 1)
students_kbins.sample(10)


# gets unique val for binned scores
students_kbins["Binned_Score"].unique()


# gets the bin edges(edges that make up the bin)
marks_edges = k_bins.bin_edges_[0]
hours_edges = k_bins.bin_edges_[1]

marks_edges, hours_edges


# in data, student where Bin score get_ipython().getoutput("= Bin hours are outliers")
# op gets them to see
students_kbins[(students_kbins["Binned_Score"] get_ipython().getoutput("= students_kbins["Binned_Hours"])].sample(10)")


# creates a comment col that will be later used to tag outlier students
students_kbins["Comment"] = ' '
students_kbins.head()


# finds rows where binn score- bin hours == 2 and insert 'suspicious' in comment col
students_kbins.loc[students_kbins["Binned_Score"] - students_kbins["Binned_Hours"] == 2, "Comment"] = "Suspicious"

students_kbins.sample(10)


students_kbins.loc[students_kbins["Binned_Hours"] - students_kbins["Binned_Score"] == 2, "Comment"] = "Needs_help"

students_kbins.sample(10)


# gets unique val from comment col
categories = students_kbins["Comment"].unique()

categories


fig, ax = plt.subplots(figsize = (12, 8))

colors = {categories[0]: "green", categories[1]: "red", categories[2]: "blue"}

ax.scatter(students_kbins["Hours Studied"], students_kbins["Score Obtained"],
           c = students_kbins["Comment"].apply(lambda x: colors[x]))

ax.set(xlabel = "Hours Studied", ylabel = "Score Obtained", title = "Students performance")

ax.grid()
ax.set_xticks(hours_edges)
ax.set_yticks(marks_edges)

plt.show()

































































































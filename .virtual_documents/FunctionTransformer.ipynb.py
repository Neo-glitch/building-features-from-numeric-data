import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import FunctionTransformer


# dummy dataset
arr = np.array([[0, -1],
               [-6, 7]])

arr


abs_transformer = FunctionTransformer(func = np.abs, 
                                      validate=False) # if true converts input to 2d mat b4 transformation
abs_transformer.transform(arr)


# fun to be used with FnTransformer
def calculate_squares(x):
    return x * x


squares_transformer = FunctionTransformer(calculate_squares, validate = False)
squares_arr = squares_transformer.transform(arr)

squares_arr











































































































































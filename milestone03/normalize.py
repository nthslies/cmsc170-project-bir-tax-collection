# imports
import numpy as np
from evaluation import get_months

# -- FOR BEHAVIORAL (OwO)
# number of region columns at the start of the features and number of month columns that replace month_sin and month_cos
REGION_COLS = 3
MONTH_COLS = 12

# function for normalizing features between 0 and 1
# the minimum and maximum come from the training data only, so test data is never used to fit
def normalize_data(x_train, x_test):
    # gets minimum and maximum for each feature
    x_min = np.min(x_train, axis = 0)
    x_max = np.max(x_train, axis = 0)
    x_range = x_max - x_min
    # handles division by zero
    x_range[x_range == 0] = 1.0
    # normalize train and test with the same training minimum and range
    x_train = (x_train - x_min) / x_range
    x_test = (x_test - x_min) / x_range

    return x_train, x_test

# -- Ibalik ntin to sin and cos
def add_month_onehot(x, sin_col = 3, cos_col = 4):
    months = get_months(x, sin_col, cos_col)

    rows = []
    for i in range(len(x)):
        # all 0 except the month of this row
        month_features = [0.0] * MONTH_COLS
        month_features[months[i] - 1] = 1.0
        # keeps the columns before month_sin and after month_cos
        row = list(x[i][:sin_col]) + month_features + list(x[i][cos_col + 1:])
        rows.append(row)

    return np.array(rows, dtype = float)

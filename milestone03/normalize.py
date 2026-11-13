# imports
import numpy as np

# function for normalizing features between 0 and 1
# the minimum and maximum come from the training data only, so the test data is never used to fit
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
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]

    return pear_value


def mse(y, pred):
    err = mean_squared_error(y, pred)
    return err


def mae(y, pred):
    err = mean_absolute_error(y, pred)
    return err


def rmse(y, pred):
    err = np.sqrt(mean_squared_error(y, pred))
    return err

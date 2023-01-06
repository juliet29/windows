import pandas as pd
import math
import numpy as np
from numpy import mean
from numpy import var
from math import sqrt


def make_df_with_freq(df, freq):
    # TODO assert that datatypes of the passed in df are as follows
    """
    takes in a dataframe with columns of the following:
    DateTime         object
    Temp C          float64
    RH %            float64
    Room              int64
    Ambient Temp    float64
    Ambient RH      float64
    Window Open       int64
    T_Delta         float64
    RH_Delta        float64
    -> this is the df that emerges from "../constants/td_ambient_102022.csv"
    and groups based on a passed in frequency signature 
    "60T", "30T", "15T", "10T", "5T", "1T", "30s"
    "30T" = 30 mins, "30s" = 30 seconds 
"""
    dftime = df.copy()

    # transform datetime index to actual pandas datetime, and set as index of dataframe 
    dftime["DateTime"] = pd.to_datetime(dftime["DateTime"])
    dftime.set_index("DateTime", inplace=True)

    # resample the dataframe to group by the correct index and drop nans
    dftime_freq = dftime.groupby("Room").resample(freq).mean()
    dftime_freq.dropna(inplace=True)

    # map window variables to 0 or 1
    dftime_freq["Window Open"] = dftime_freq["Window Open"].round()

    return dftime_freq


def str2dt(date):
    """Convert a string to a pandas timestamp object """
    return pd.to_datetime(date, format= '%Y, %m, %d, %H, %M' )

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def normalize(arr):
    arr2 =  (arr - arr.min())/(arr.max() - arr.min()) 
    return arr2


def rmse(arr1, arr2):
    MSE = np.square(np.subtract(arr1, arr2)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

import pandas as pd
import math
import numpy as np
from numpy import mean
from numpy import var
from math import sqrt
import json
import plotly.graph_objects as go
import scipy.optimize

## typical imports 



# Key for categorical data in constants/td_ambient_102022.csv
# Room: 442A = 0, 422B = 1
# Window Open: Closed = 0, Open = 1





## ------------------------- ! Tools 
def str2dt(date):
    """Convert a string to a pandas timestamp object 
    Format should be: '2022, 07, 24, 07, 20' """
    return pd.to_datetime(date, format= '%Y, %m, %d, %H, %M' )

def check_window_treatment(df):
    window_check = df["Window Open"].unique()

    if len(window_check) >= 2:
        window_treatment = "Variable Window"
    else:
        window_treatment = "Constant Open Window" if window_check[0] == 1 else "Constant Closed Window"
    return window_treatment

def find_nearest(array, value):
    "find index of closest value in an array to a value that has passed in "
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

colorway = ['#702632', '#A4B494', '#495867', '#912F40', "#81909E", "#F4442E", "#DB7C26", "#BB9BB0"]



## ------------------------- ! End Tools 

# 

# 

## ------------------------- ! Pre-Processing  

def normalize(arr):
    "normalize the values in an array to be between 0 and 1 based on the minimum and maximum values in the array"
    arr2 =  (arr - arr.min())/(arr.max() - arr.min()) 
    return arr2

def normalize_scale(arr, tmin, tmax):
    "normalize the values in an array to be between 0 and 1 based on the minimum and maximum values in the array"
    arr2 =  (arr - arr.min())/(arr.max() - arr.min())  * (tmax - tmin) + tmin
    return arr2

def take_derivative(series):
    return pd.Series(normalize(np.gradient(series)))
## ------------------------- ! End Pre-Processing  

# 

# 

## ------------------------- ! Metrics 
def rmse(arr1, arr2):
    "calculate the root mean squared error between two arrays"
    MSE = np.square(np.subtract(arr1, arr2)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

def mbe(true, pred):
    "calculate the mean bias error between two arrays"
    mbe_loss = np.mean(true - pred)
    return mbe_loss

def cohend(d1, d2):
    """ function to calculate Cohen's d for independent samples """
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

def calc_zscore(arr):
    z_score = np.apply_along_axis(lambda x, mean, std: (x - mean)/std, 0, arr, arr.mean(), arr.std())
    return z_score

def calc_win_change_dist(df, ix):
    """
    Report if the index (ix) returned by an algorithm is what the data (df) recognizes as a window flip 

    Returns
    exact: if ix reported by the algo is spot on 
    nearest: the nearest ix to what was reported 
    distance: the distance (# of indices) between the ix and the nearest found value 
    """
    # note where the value in Window Open series changes 
    shift  = df["Window Open"].shift() != df["Window Open"]

    # return if ix > lenght of data 
    if ix > len(shift):
       return "Passed index larger than it should be"
    
    # some of the experiments did not have any shifts 
    flips = np.where(shift*1==1)[0] #[1:-1]
    if len(flips) < 2:
        return "No Flip Happened!"
    
    # check if got the exact flip 
    exact = shift[ix]
    
    # drop the first entry which will always be true 
    flips = flips[1:-1]

    # find the distance between the nearest flip, and the ix 
    nearest = find_nearest(flips, ix)
    distance = find_nearest(flips, ix) - ix

    # report performance 
    return exact, nearest, distance 

## ------------------------- ! End Metrics 

# 

# 

## ------------------------- ! Reorganize/Import Data 

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
    See classification_102022 for use case
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


def day_split(arr_of_dfs):
    """splits all the dfs in the arr_of_dfs list into smaller dataframes that are split by day 
    used in the daily_analysis notebooks 
    """
    arr_of_split_dfs = []
    for arr in arr_of_dfs:
        # dataframe might have timing information in the index, or in a column so try out both 
        try:
            daysplit_list = [group[1] for group in arr.groupby(arr.index.date)]
        except:
            daysplit_list = [group[1] for group in arr.groupby(arr["DateTime"].dt.date)]
        arr_of_split_dfs.append(daysplit_list)
        # print(daysplit_list)

    return arr_of_split_dfs

def import_desired_data(exp, freq):
    """
    get specific experiment, specific room, and specific frequency of data 

    exp: exp month, A, B, C -> july, aug, sep 
    freq: "60T" = 60 mins, "0.5T" = 1/2 minute

    returns two dataframes split into room 1 and room 0, and optionally for a given experiment, and a given frequency 
    """
    
    # read in the dataframe 
    try:
        df = pd.read_csv("../constants/td_ambient_102022.csv" )
    except: 
        df = pd.read_csv("/Users/julietnwagwuume-ezeoke/_UILCode/windows/analysis/constants/td_ambient_102022.csv" )
    # transform datetime index to actual pandas datetime, 
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    if exp != None:
        # select certain experiments 
        try:
            with open('../constants/window_treatment.json') as f:    
                window_treatment = json.load(f)
        except:
            with open('/Users/julietnwagwuume-ezeoke/_UILCode/windows/analysis/constants/window_treatment.json') as f:    
                window_treatment = json.load(f)

        exp_a_end = str2dt(window_treatment["072522"]["cutoff_times"]["end"])
        exp_b_end = str2dt(window_treatment["081622"]["cutoff_times"]["end"])

        # HERE branching logic bassed on which experiment considering 
        if exp == "A":
            df_exp = df.loc[df["DateTime"] < exp_a_end]
        elif exp == "B":
            mask = (df['DateTime'] > exp_a_end) & (df['DateTime'] < exp_b_end)
            df_exp = df.loc[mask].reset_index(drop=True)
        elif exp == "C":
            df_exp = df.loc[df["DateTime"] > exp_b_end]
        else:
            df_exp = None

        # split data into dataframes based on room
        try:
            df1, df0 = [x.reset_index(drop=True) for _, x in df_exp.groupby(df_exp['Room'] < 1)]
        except:
            "Invalid experiment name entered! - Try 'A', 'B', or 'C' "
            pass

    else:
        df1, df0 = [x.reset_index(drop=True) for _, x in df.groupby(df['Room'] < 1)]

    if freq:
        # filter data based on frequency 
        df00 = make_df_with_freq(df0, freq).droplevel(0).reset_index()
        df01 = make_df_with_freq(df1, freq).droplevel(0).reset_index()

        return df00, df01 
    else:
        return df0, df1

## ------------------------- ! End Reorganize Data 

# 

# 

## ------------------------- ! Plots 

def temperature_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["DateTime"],
        y=df["Temp C"], 
        mode='lines+markers',
    ))

    fig.update_layout(xaxis_title='Dates',
                    yaxis_title='Temperature (ºC)',
                    title=f"Room {df['Room'][0]} - {check_window_treatment(df)}")
    return fig


def plot_many(fig, time, series, names, ):
    for name, ser in zip(names, series):
            fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode='lines'))
    return fig


## ------------------------- ! End Plots 


## ------------------------- ! Miscellaneous  

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"

    # https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    
    '''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

## ------------------------- !End  Miscellaneous  
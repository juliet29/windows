import plotly.graph_objects as go
# import kaleido
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import json

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import scipy.optimize

import sys
sys.path.insert(0, "../scripts")
import helpers as h
import seaborn as sns


# ============================================================================ #
# Smoothing functions 

def make_stl_smooth(series, time, model="additive", ):
    # TODO make flexible for period 
    n_samples = 4 # 15 minute intervals (4/hour)
    #n_samples = 12 # 5 minute intervals (4/hour)
    seasonality_period = 24 # 24 hour temperature period 
    period = n_samples*seasonality_period
    result = seasonal_decompose(series, model=model, period=period)
    return result.seasonal

def make_sin_smooth(series, time):
    fit = h.fit_sin(time, series)
    return fit["fitfunc"](time)

def make_ewm_smooth(series, time, level=4):
    return series.ewm(level).mean()





# ============================================================================ #
# Window detection class 

class Window_Detect2:
    def __init__(self, df):
        self.time = df["DateTime"]
        # magnitude of time passed in seconds 
        time_mag = self.time - self.time.min()
        self.time_seconds = time_mag.dt.total_seconds()

        self.window = df["Window Open"]
        self.window_norm = h.normalize(self.window)

        self.temp = df["Temp C"]
        self.temp_norm = h.normalize(self.temp)


    def analyze_window_change(self, smooth_fx, sim_smooth=None):
        if sim_smooth is not None:
            self.smooth_series = h.normalize(sim_smooth)
        else:
            self.smooth_series = h.normalize(smooth_fx(self.temp_norm, self.time_seconds))

        # TODO: mean largest - mean smallest?
        self.dif = h.normalize(self.temp_norm - self.smooth_series)
        self.deriv = h.take_derivative(self.dif)
        self.deriv2 = h.take_derivative(self.deriv)
        self.std, self.std2 = self.deriv.std(), self.deriv2.std()
        self.zscore, self.zscore2 = h.calc_zscore(self.deriv), h.calc_zscore(self.deriv2)


    def plot_analysis(self):
        fig = go.Figure()

        series = [self.window_norm, self.temp_norm, self.smooth_series, self.dif, self.deriv, self.deriv2]
        names = ["Window", "Observed Temp", "Smoothed", "Difference", "Deriv1", "Deriv2"]

        return h.plot_many(fig, self.time, series, names)


    def plot_distributions(self, marker_width=0.1, bin_size=0.003):
        print(f"Std 1 = {self.std}, Std 2 = {self.std2}")
        fig = go.Figure()

        for ix, ser in enumerate([self.deriv2, self.deriv]):
            opacity = 0.9 if ix == 0 else 1
            fig.add_trace(go.Histogram(
            x=ser, histnorm='probability', name=f' Deriv{2 - ix}', opacity=opacity, marker_line=dict(width=marker_width ,color='black'), xbins=dict( size=bin_size),))

        fig.update_layout(barmode="stack")

        return fig
        

    def plot_zscore(self):
        fig = go.Figure()

        series = [self.window_norm, self.zscore, self.zscore2]
        names = ["Window", "Z-Score 1", "Z-Score 2"]

        return h.plot_many(fig, self.time, series, names)


    def plot_guesses(self, timedelta=15*2):
        # TODO change to make and plot or split function
        guess_mask = (self.zscore > 2) | (self.zscore <= -2)
        guess_times = self.time[guess_mask]

        # remove guesses that are due to bouncing
        clean_mask = guess_times.diff() >= pd.Timedelta(minutes=timedelta) # TODO make this a function of the time lag in the dataframe 
        self.guess_times = guess_times[clean_mask]

        # plot guesses in terms of zscore2
        self.zscore2_norm = h.normalize(self.zscore2)
        self.guess_values = self.zscore2_norm[self.guess_times.index] 

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time, y=self.window_norm, name="Window", mode='lines'))
        fig.add_trace(go.Scatter(x=self.guess_times, y=self.guess_values, name="Guess ~ Z-Score 2", mode='markers'))

        return fig


    def plot_analysis_and_distributions(self):
        # TODO make into subplots fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig1 = self.plot_analysis()
        fig2 = self.plot_distributions()
        fig1.show()
        fig2.show()
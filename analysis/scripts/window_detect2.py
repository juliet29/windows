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
# Window detect specific plotting functions that operacte across objects 

def make_dual_plot(time, obj1, obj2, title_arr): 
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(title_arr[0], title_arr[1],))

    s2, s3 = [[obj.window_norm, obj.temp_norm, obj.smooth_series, obj.dif, obj.deriv, obj.deriv2] for obj in [obj1, obj2]]

    names = ["Window", "Observed Temp", "Smoothed", "Difference", "Deriv1", "Deriv2"]
    
    opacities = [0.4] + [1]*(len(s2)-1)

    for ix, name, ser in zip(range(len(names)), names, s2):
        fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode='lines', line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name,), row = 1, col = 1)

    for ix, name, ser in zip(range(len(names)), names, s3):
        fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode='lines', line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name, showlegend=False), row = 1, col = 2)

    return fig, names

def update_dual_plot(fig, names, show_arr, ):
    for name in names: 
        if name in show_arr:
            fig.update_traces(visible=True, selector=dict(name=name))
        else:
            fig.update_traces(visible="legendonly", selector=dict(name=name))
    return fig



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

    def make_guesses(self, timedelta=15*2, z=1):
        # normalized values for plaotting 
        self.zscore_norm = h.normalize(self.zscore)
        self.zscore2_norm = h.normalize(self.zscore2)

        if z==1:
            zscore = self.zscore
            zscore_norm = self.zscore_norm
        else:
            zscore = self.zscore2
            zscore_norm = self.zscore2_norm


        guess_mask = (zscore >= 2) | (zscore <= -2)
        guess_times = self.time[guess_mask]

        # remove guesses that are due to bouncing
        clean_mask = guess_times.diff() >= pd.Timedelta(minutes=timedelta) 
        self.guess_times = guess_times[clean_mask]

        
        self.guess_values = zscore_norm[self.guess_times.index] 

        


    def analyze_window_change(self, smooth_fx, sim_smooth=None, timedelta=15):
        if sim_smooth is not None:
            self.smooth_series = h.normalize(sim_smooth)
        else:
            self.smooth_series = h.normalize(smooth_fx(self.temp_norm, self.time_seconds))

        self.dif = h.normalize(self.temp_norm - self.smooth_series)
        self.deriv = h.take_derivative(self.dif)
        self.deriv2 = h.take_derivative(self.deriv)
        self.std, self.std2 = self.deriv.std(), self.deriv2.std()
        self.zscore, self.zscore2 = h.calc_zscore(self.deriv), h.calc_zscore(self.deriv2)

        self.make_guesses(timedelta)


    def plot_analysis(self):
        fig = go.Figure()

        series = [self.window_norm, self.temp_norm, self.smooth_series, self.dif, self.deriv, self.deriv2]
        names = ["Window", "Observed Temp", "Smoothed", "Difference", "Deriv1", "Deriv2"]

        return h.plot_many(fig, self.time, series, names)


    def plot_distributions(self, marker_width=0.1, bin_size=0.003):
        # print(f"Std 1 = {self.std}, Std 2 = {self.std2}")
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
        self.make_guesses(timedelta)

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
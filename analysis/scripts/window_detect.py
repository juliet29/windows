import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import json


from statsmodels.tsa.seasonal import seasonal_decompose

# local modules 
import sys
sys.path.insert(0, "../scripts")
import helpers as h


class Window_Detect:
    def __init__(self, exp, period="15T"):
        self.exp = exp 
        self.period = "15T"

    def determine_varied_room(self):
        "only looking at the room that had variation really.."
        room0, room1 = h.import_desired_data(self.exp, self.period)

        varied_room = room0 if len(room0["Window Open"].unique()) > 1 else room1
        
        self.varied_room = varied_room.set_index(varied_room["DateTime"].values)
        assert len(self.varied_room["Window Open"].unique()) > 1

        return self.varied_room
    

    def make_s3d(self):
        "s3d = stl_deriv_dif_deriv"

        # ~ calculate derivative of observation 
        self.obs_deriv = h.normalize(pd.Series(np.gradient(self.varied_room["Temp C"]), self.varied_room.index, name='obs_deriv'))
        
        # ~ seasonal decomposition of the derivative of the observation 
        # period => # samples per unit in length of time over which seasonality occurs (here, 4 samples/1 hour) * length of time over which seasonality occurs (here, 24 hours)
        n_samples = 4 # TODO: make this a function of the period
        seasonality_period = 24
        period = n_samples*seasonality_period
        self.stl_deriv = seasonal_decompose(self.obs_deriv,model='additive', period=period)

        # ~ difference between obeservation and the seasonal decomposition 
        self.stl_deriv_dif = h.normalize(h.normalize(self.stl_deriv.seasonal) - h.normalize(self.varied_room["Window Open"]))

        # ~ derivative of the difference 
        self.stl_deriv_dif_deriv = h.normalize(pd.Series(np.gradient(self.stl_deriv_dif), self.stl_deriv_dif.index, name='deriv'))

        return 
    
    def make_guesses(self):
        # find the points at which the derivative is greater than some threshold (here +- 0.3 from the mean) using a mask 
        s3d = pd.DataFrame(self.stl_deriv_dif_deriv)
        mask = (s3d["deriv"] > 0.8) | (s3d["deriv"] <= 0.2) #TODO somehow detect this automatically...
        m = s3d.loc[mask]

        # find where the difference in times is not equal to 15 (time perood of each data collection in this modified dataset => see h.import_deired_data())
        diff_series = pd.Series(m.index).diff() # drop where diff = 15 
        duplicate_mask = diff_series != pd.Timedelta(minutes=15)

        # make the datetime index into a column that can be accessed
        mdt = m.reset_index()
        a = mdt["index"][duplicate_mask].index
        clean_deriv = mdt.iloc[a]
        self.guess = clean_deriv

        return 
    
    def main(self):
        self.determine_varied_room()
        self.make_s3d()
        self.make_guesses()
        fig = self.plot_guesses()

        return fig

    def plot_guesses(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.varied_room["DateTime"].index,
            y=h.normalize(self.varied_room["Window Open"]), 
            mode='lines',
            name="Window Schedule",
            line=dict(width=1),
        ))
        fig.add_trace(go.Scatter(
            x = self.guess["index"],
            y=self.guess["deriv"], 
            mode='markers',
            name="Guess"
        ))

        return fig

    
    def plot_s3d(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                            x=self.varied_room["DateTime"].index,
                            y=h.normalize(self.varied_room["Window Open"]), 
                            mode='lines',
                            name="Window Sched",
                            line=dict(width=1),
                        ))

        fig.add_trace(go.Scatter(
                            x=self.varied_room["DateTime"].index,
                            y=h.normalize(self.varied_room["Temp C"]), 
                            mode='lines',
                            name="Variable Open Obs"
                        ))

        # derivative of observation only 
        fig.add_trace(go.Scatter(
                            x=self.obs_deriv.index,
                            y=self.obs_deriv, 
                            mode='lines',
                            name="Obs 1st Deriv "
                        ))

        # difference between this stl decomp 
        fig.add_trace(go.Scatter(
                            x=self.stl_deriv_dif.index,
                            y=self.stl_deriv_dif, 
                            mode='lines',
                            name="Obs STL Dif  Deriv "
                        ))
        # s3d
        fig.add_trace(go.Scatter(
                            x=self.stl_deriv_dif_deriv.index,
                            y=self.stl_deriv_dif_deriv, 
                            mode='lines',
                            name="Deriv of Obs STL Dif  Deriv "
                        ))
        return fig


    





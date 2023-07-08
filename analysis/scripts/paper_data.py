import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

import seaborn as sns
import seaborn.objects as so
# Apply the default theme
sns.set_theme()
import matplotlib.pyplot as plt



import sys
sys.path.insert(0, "../scripts")
import helpers as h
import window_detect2 as w
import scores as s

class Paper_Data: 
    def __init__(self, exp):
        self.exp = exp 
        self.window_sched = self.exp["Window Open"]
        self.datetime = self.exp["DateTime"]

    def create_in_out_df(self, col1="Temp C", col2="Ambient Temp"):
        in_val = {
            "val": self.exp[col1],
            "area": pd.Series(["in"]*len(self.exp))
        }
        out_val  = {
            "val": self.exp[col2],
            "area": ["out"]*len(self.exp)
        }

        res = pd.concat([pd.DataFrame(in_val), pd.DataFrame(out_val)])
        return res 

    def plot_distributions(self):
        temps=self.create_in_out_df()
        rhs = self.create_in_out_df("RH %", "Ambient RH" ) 

        fig, axs = plt.subplots(1, 2,figsize=(12, 3))
        sns.kdeplot(data=temps,  x="val", hue="area",  fill=True, ax=axs[0])
        axs[0].set_title("Temperature")
        sns.kdeplot(data=rhs,  x="val", hue="area",  fill=True, ax=axs[1])
        axs[1].set_title("Relative Humidity")

        self.dist_fig = fig 

        self.dist_fig.show()


    def cal_avg_window_time_open(self):
        timedelta = self.datetime[1] - self.datetime[0]

        guess_times_ix = s.identify_changed_ix(self.window_sched)

        pred_change = {change_ix: time_ix for  change_ix, time_ix in  enumerate(guess_times_ix)}

        if self.window_sched[guess_times_ix[0]] == 0:
            print("A")
            start_vals = list(pred_change.values())[1::2]
            close_vals = list(pred_change.values())[2::2]
        else:
            # print("B and C")
            start_vals = list(pred_change.values())[0::2]
            close_vals = list(pred_change.values())[1::2]

        ixes = [(i,j) for i,j in zip(start_vals, close_vals)]


        lens = []
        for i in range(len(ixes)):
            window_vals = self.window_sched[ixes[i][0]:ixes[i][1]]
            assert window_vals.all() # all the values should be 1 
            lens.append(len(window_vals))
            
        lens = np.array(lens)

        avg_len = np.mean(lens)
        median_len = np.median(lens)
        
        self.avg_len_time = avg_len*timedelta
        self.median_len_time = median_len*timedelta


    def calc_opening_percentage(self):
        win_open_group =  self.exp.groupby("Window Open").count()
        
        total_time_periods = win_open_group.iloc[1]["DateTime"] + win_open_group.iloc[0]["DateTime"]
        
        open_time_periods = win_open_group.iloc[0]["DateTime"]

        self.perc_time_open = open_time_periods/total_time_periods

    def calc_table_data(self):
        self.calc_opening_percentage()
        self.cal_avg_window_time_open()

        self.data = {
            "Starting Day": self.datetime[0],
            "Data Length": self.datetime.iloc[-1] - self.datetime[0],
            "Room": self.exp["Room"][0],
            "Opening Percentage": self.perc_time_open,
            # "Average Open Time": self.avg_len_time
            "Median Open Time": self.median_len_time
        }

        return self.data 
""" Contains Temp_RH_Analyis module (mostly for visualization though) -> created 08/23/22 """

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import OrderedDict
import os
import json
import datetime

class Temp_RH_Analysis:
    """Analysis for sets of data collected within the same time frame"""
    def __init__(self, root, date, room_labels):
        self.root = root  # root should go all the way up to temp_rh
        self.date = date # date ending time period of data collection
        self.room_labels = room_labels # array of two strings describing window treatment in rooms A and B
        
        # get the data for that set of days, and save it in dataframes
        self.parse_room_data(date)

        # init all other variables to none 
        self.cutoff_time = None
        self.highlight_times = None
        self.ambient_data = None
        self.room_data_comp_fig = None


    
    def parse_trh_csv(self, csv):
        df = pd.read_csv(csv, header=0, names=["DateTime", "Temp C", "RH %"], usecols=[1,2,3], skiprows=[1])
        # convert string dates to datetime format 
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        return df 


    def parse_room_data(self, date):
        """read data from csv for rooms a and b, within given date range"""
        trh_422a_csv = os.path.join(self.root, f"data/hobo_temprh/TRH_4_422A_{date}.csv")
        self.trh_422a = self.parse_trh_csv(trh_422a_csv)

        trh_422b_csv = os.path.join(self.root, f"data/hobo_temprh/TRH_2_422B_{date}.csv")
        self.trh_422b = self.parse_trh_csv(trh_422b_csv)

    def plot_trh_for_room(self):
        # TODO 
        "hello"

    def string_to_datetime(self, date):
        """Convert a string to a pandas timestamp object """
        return pd.to_datetime(date, format= '%Y, %m, %d, %H, %M' )


    def make_highlight_times(self, y_vals=[21,28]):
        with open('../constants/htimes.json') as f:    
            htimes = json.load(f)
        if "Closed" in self.room_labels:
            h = htimes[self.date]
            htimes_arr = []
            for i in  range(len(h)-1):
                # TODO figure out why calling df here...
                a = self.string_to_datetime(h[i]["close"])
                b = self.string_to_datetime(h[i+1]["open"])
                htimes_arr.append((a,b))
        else:
            htimes_arr = [(self.string_to_datetime(i["open"]), self.string_to_datetime(i["close"]) ) for i in  htimes[self.date]]

        my_shapes = []
        for val in htimes_arr:
            my_shapes.append(dict(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=val[0],
                y0=y_vals[0],
                x1=val[1],
                y1=y_vals[1],
                fillcolor="lightgray",
                opacity=0.4,
                line_width=0,
                layer="below"
            )))
        self.highlight_times = my_shapes


    def make_cutoff_times(self, c_start, c_end):
        # c_start / c_end = cutoff data array of month, day, hour, minute, year 
        starttime = pd.Timestamp(2022, c_start[0], c_start[1], c_start[2], c_start[3], c_start[4])
        endtime = pd.Timestamp(2022,  c_end[0], c_end[1], c_end[2], c_end[3], c_end[4])
        self.cutoff_time = [starttime, endtime] 

    def far2cel(self, fnum):
        """farenheit to celcius converter"""
        return (fnum - 32) * (5/9)

    def get_ambient_data(self):
        su_csv = os.path.join(self.root, f"data/stanford_weather_data/SU_Hourly_{self.date}.csv") 
        su_df = pd.read_csv(su_csv, header=0, names=["Date", "Time", "Temp F"], usecols=[0,1,2], skiprows=[1])

        # adjust date and time data 
        dates = pd.to_datetime(su_df["Date"])
        times = pd.to_datetime(su_df["Time"]/100, format="%H").dt.time
        fulldatetimes = [datetime.datetime.combine(d, t) for d, t in zip(dates, times)]
        su_df["datetimes"] = pd.to_datetime(fulldatetimes)

        # sort by dataframe by date 
        su_df_sort = su_df.sort_values("datetimes")
        hdata_su_c = self.far2cel(su_df_sort["Temp F"])
        self.ambient_data = {"times": su_df_sort["datetimes"], "temps": hdata_su_c}


    
    def plot_room_data_comparison(self):
        # base comparison plot
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=self.trh_422a["DateTime"], y=self.trh_422a["Temp C"],
                            mode='lines',
                            name=f'422A - {self.room_labels[0]} '))
        fig.add_trace(go.Scatter(x=self.trh_422b["DateTime"], y=self.trh_422b["Temp C"],
                            mode='lines',
                            name=f'422B - {self.room_labels[1]} '))

        fig.update_yaxes(title_text="Temperature ºC", secondary_y=False)
        # fig.update_yaxes(title_text="Relative Humidity %", secondary_y=True)

        # add ons 


        if self.cutoff_time:
            fig.update_xaxes(range=[self.cutoff_time[0], self.cutoff_time[1]], row=1, col=1)
            # work out the title using the cutoff times 
            start = self.cutoff_time[0].date().strftime("%m/%d")
            end  = self.cutoff_time[1].date().strftime("%m/%d")
            title_text=f"Temperature Comparison {start} - {end} "
        else: 
            title_text=f"Temperature Comparison, Data Processed on {self.date}"
        
        fig.update_layout(title_text=title_text)

        if self.highlight_times:
            fig.update_layout(
                shapes=self.highlight_times
            )
        
        if self.ambient_data:
            fig.add_trace(go.Scatter(x=self.ambient_data["times"], y=self.ambient_data["temps"],
                    mode='lines',
                    name='Ambient'))
        

        self.room_data_comp_fig = fig
        
        fig.show()
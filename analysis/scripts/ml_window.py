import plotly.graph_objects as go
import plotly.express as px
# import kaleido
from plotly.subplots import make_subplots
import seaborn as sns
import plotly.io as pio
import pandas as pd
import numpy as np
import json

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import scipy.optimize 
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, "../scripts")
import helpers as h
import window_detect2 as w


class ML_Window_Detect:
    
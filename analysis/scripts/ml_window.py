import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report


class ML_Window_Detect:
    """ pass in numpy arrays """
    def __init__(self, ts_arr, truth):
        self.ts_arr = ts_arr
        self.truth = truth 

    
    def create_features(self):
        if len(self.ts_arr) == 1:
            self.x = self.ts_arr[0] #.to_numpy()
        elif len(self.ts_arr) == 2:
            self.x = np.array(self.ts_arr).T

        return 

    
    def fit_and_decide(self):
        self.model = OneClassSVM().fit(self.x)
        self.choices = self.model.decision_function(self.x)
        self.standard_metrics = classification_report(self.truth, self.choices)

        return 

    def run_all(self):
        self.create_features()
        self.fit_and_decide()

        return 
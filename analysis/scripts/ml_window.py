import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report


class ML_Window_Detect:
    
    def __init__(self, ts_arr, truth):
        """
            ts_arr: has to be an arry of numpy arrays
            truth: numpy array of true value -> window open schedule 
        """
        self.ts_arr = ts_arr
        self.truth = truth 

    
    def create_features(self):
        if len(self.ts_arr) == 1:
            self.x = self.ts_arr[0].reshape(-1,1) #.to_numpy()
        elif len(self.ts_arr) >= 2:
            self.x = np.array(self.ts_arr).T

        return 

    
    def fit_and_decide(self):
        self.model = OneClassSVM().fit(self.x)
        self.choices = self.model.decision_function(self.x)
        # map probabilities / mll given by the decision function to 1 (window open) or 0 
        self.choices  = np.array([1 if i >= 0 else 0 for i in self.choices])

        # compute metrics
        self.standard_metrics_str = classification_report(self.truth, self.choices)
        self.standard_metrics = classification_report(self.truth, self.choices, output_dict=True)
        self.accuracy = self.standard_metrics["accuracy"]

        return 

    def run_all(self):
        self.create_features()
        self.fit_and_decide()

        return
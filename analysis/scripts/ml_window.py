import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, "../scripts")
import scores as s
import helpers as h

from icecream import ic 
import copy


def create_features_arr(arr):
        if len(arr) == 1:
            x = arr[0].reshape(-1,1) #.to_numpy()
        elif len(arr) >= 2:
            x = np.array(arr).T

        return x


class ML_Window_Detect_Many: 
    def __init__(self, exp, test_set):
        
        self.exp = exp 
        self.truth = self.exp["Window Open"]
        self.test_set = test_set

        # with open(f'../constants/{test_set_path}.json','w') as f:
        #     self.test_set = json.load(f)

    def create_data_dict(self):
        arrs = {}
        # measured data 
        arrs["meas_temp"] = self.exp["Temp C"]
        arrs["meas_rh"] = self.exp["RH %"]

        # ambient data 
        arrs["amb_temp"] = self.exp["Ambient Temp"]
        arrs["amb_rh"] = self.exp["Ambient RH"]

        arrs = {k: np.array(h.normalize(v)) for k,v in arrs.items()}

        # derivatives
        derivs = {f"dt_{k}": h.normalize(np.gradient(v)) for k,v in arrs.items()}

        # meausred/ambient temp difference 
        diffs = {}
        diffs["amb_minus_meas_temp"] = arrs["amb_temp"] - arrs["meas_temp"]
        diffs["amb_minus_meas_rh"] = arrs["amb_rh"] - arrs["meas_rh"]

        diffs["meas_minus_deriv_temp"] = arrs["meas_temp"] - derivs["dt_meas_temp"]
        diffs["meas_minus_deriv_rh"] = arrs["meas_rh"] - derivs["dt_meas_rh"]

        diffs = {k: h.normalize(v) for k,v in diffs.items()}

        # all data 
        self.all_data = arrs | derivs | diffs


    def fit_and_score_many(self):
        self.all_results = {}
        for ix, set in enumerate(self.test_set):
            self.all_results[ix] = {}
            self.all_results[ix]["name"] = set
            # ic(self.all_results[ix]["name"])

            # pull the data and run the model 
            set_data = []
            for item in set:
                set_data.append(self.all_data[item])

            features = create_features_arr(set_data)
            model = OneClassSVM().fit(features)
            choices = model.decision_function(features)
            choices = pd.Series([1 if i >= 0 else 0 for i in choices])

            s1 = s.Scores(exp=self.exp, choices=choices)
            s1.calc_all_metrics()
            self.all_results[ix]["metrics"] =  s1.short_metrics 


            self.complete_results = copy.deepcopy(self.all_results)
            self.complete_results[ix]["metrics"] = 2

        return self.all_results

    
    def sort_by_performance(self, type="f1"):
        if type=="f1":
            self.sorted_results = sorted(self.all_results.items(), key=lambda x:x[1]["metrics"]["standard"]["macro avg f1-score"], reverse=True)
            self.quick_res = [
            {
                "name": v["name"],
                "val": v["metrics"]["standard"]["macro avg f1-score"] 
            }
            for k,v in self.sorted_results]
        elif type=="ub_acc":
            self.sorted_results = sorted(self.all_results.items(), key=lambda x:x[1]["metrics"]["drdr"]["unbounded acc"], reverse=True)
            self.quick_res = {k:v["metrics"]["drdr"]["unbounded acc"] for k,v in self.sorted_results}
        else:
            self.sorted_results = self.all_results

        return self.sorted_results
        
        








class ML_Window_Detect:
    
    def __init__(self, ts_arr, truth, exp, test_set_path="test_set_230703"):
        """
            ts_arr: has to be an arry of numpy arrays
            truth: numpy array of true value -> window open schedule 
        """
        self.ts_arr = ts_arr
        self.truth = truth 
        self.exp = exp 


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
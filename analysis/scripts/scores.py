import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "../scripts")
import helpers as h

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

from itertools import groupby

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    len_intersect = len(a_set.intersection(b_set))
    if len_intersect > 0:
        return True, len_intersect
    return False, 0 


class Scores:
    def __init__(self, exp, choices, guess_times, near_miss_lim=2, ):
        """
        exp: a dataframe related to one of the experiments, result of h.import_desired_data  

        choices: numerically indexed *Series* with 0s and 1s 
        guess_times: (non-consecutive) numerically indexed *Series* with DateTimes that reflect times when predict window state changes 
        
        """
        self.near_miss_lim = near_miss_lim
        self.exp = exp
        self.timedelta = self.exp["DateTime"][1] - self.exp["DateTime"][0]
        self.truth = self.exp["Window Open"]

        self.choices = choices
        self.guess_times = guess_times 



    def calc_hit_scores(self):
        shift = self.exp["Window Open"].shift() != self.exp["Window Open"]
        true_change_indices = np.where(shift==True)[0]

        true_change = {change_ix: time_ix for  change_ix, time_ix in  enumerate(true_change_indices)}

        pred_change = {change_ix: time_ix for  change_ix, time_ix in  enumerate(self.guess_times.index)}

        res = {}
        for k, v in pred_change.items():
            nearest_ix = h.find_nearest(list(true_change.values()), v)
            distance = v - nearest_ix
            res[k] = {}
            res[k]["match?"] = True if distance == 0 else False
            res[k]["nearest_ix"] = nearest_ix
            res[k]["distance"] = distance

        
        self.scores = {
            "hits": 0, "near_hits": 0, "miss": 0
        }

        for k, v in res.items():
            if v["match?"]:
                self.scores["hits"]+=1
            elif v["distance"] <=2:
                self.scores["near_hits"]+=1
            else:
                self.scores["miss"]+=1

        # calculate ratios 
        self.nice_results = {
            "hits/guesses": self.scores["hits"] / len(pred_change),

            "hits/actions": self.scores["hits"]/len(true_change),

            "(hits + near hits)/guesses": (self.scores["hits"] +  self.scores["near_hits"])/ len(pred_change),

            "(hits + near hits)/actions": (self.scores["hits"] +  self.scores["near_hits"])/len(true_change),

            "misses/guesses": self.scores["miss"]/ len(pred_change),
        }

        self.nice_results = {k:np.round(v,3) for k,v in self.nice_results.items()}

        # add values that are not ratios 
        self.nice_results.update({
            "number of actions": len(true_change),
            "number of guesses": len(pred_change)
        })

        # join the nice results and simple score sums in one dictionary 
        self.nice_results.update(self.scores)

        self.nice_res_df = pd.DataFrame.from_dict(self.nice_results, orient="index", columns=["results"])

        return  self.nice_res_df


        


    def calc_win_change_dist(self, df, ix):
        # ensure index is within the length of the data 
        assert ix <= len(df["Window Open"]) 

        # ensure that the changes in window state where actually observed, should have 0s and 1s 
        assert len(df["Window Open"].unique()) > 1 


        # note where the value in Window Open series changes 
        shift  = df["Window Open"].shift() != df["Window Open"] # boolean series 
        flips = np.where(shift*1==1)[0] #[1:-1]
        
        # check if got the exact flip, this returns a bool 
        exact = shift[ix]
        
        # drop the first entry which will always be true 
        flips = flips[1:-1]

        # find the distance between the nearest flip, and the ix 
        nearest = h.find_nearest(flips, ix)
        distance = h.find_nearest(flips, ix) - ix

        # report performance 
        return exact, nearest, distance 


    def count_score(self, c):
        if c == 0:
            self.hits+=1
        elif c <= self.near_miss_lim:
            self.near_miss+=1
        else:
            self.miss+=1
        
        return self.hits, self.near_miss, self.miss


    def report_scores(self):
        # guess tims are the specific times at which guesses have been made 
        self.hits = 0 
        self.near_miss = 0 
        self.miss = 0

        self.num_actions = len(self.exp[self.truth.shift() != self.truth])

        # see how far guess times are from actual times of window schedule change 
        df = pd.DataFrame(self.guess_times)
         
        self.win_distances = df.apply(lambda x: self.calc_win_change_dist(self.exp, x.name), axis=1)

        # calculate scores and sum up 
        self.scores = self.res.apply(lambda x: self.count_score(np.abs(x[2]))) # TODO change to use name in df instead of index (nearest)
        self.score_sum = pd.DataFrame(self.scores.iloc[-1], index=["hit", "near_hit", "miss"]).T

        # calculate ratios 
        self.nice_results = {
            "hits/guesses": self.score_sum["hit"][0] / len(self.scores),

            "hits/actions": self.score_sum["hit"][0]/self.num_actions,

            "(hits + near hits)/guesses": (self.score_sum["hit"][0] +  self.score_sum["near_hit"][0])/ len(self.scores),

            "(hits + near hits)/actions": (self.score_sum["hit"][0] +  self.score_sum["near_hit"][0])/self.num_actions,

            "misses/guesses": self.score_sum["miss"][0]/ len(self.scores),
        }

        # convert ratios into percentages
        self.nice_results = {k:1*np.round(v,3) for k,v in self.nice_results.items()}

        # add values that are not ratios 
        self.nice_results.update({
            "number of actions": self.num_actions,
            "number of guesses": len(self.scores)
        })

        # join the nice results and simple score sums in one dictionary 
        self.nice_results.update(self.score_sum.T.to_dict()[0])

        self.nice_res_df = pd.DataFrame.from_dict(self.nice_results, orient="index", columns=["results"])

        return  self.nice_res_df

    
# De Rautlin de Roy metrics 
    
    def determine_openings(self, series):

        df = series.to_frame(name="Name")
        grp = df.groupby("Name")

        # openings have a value of 1, so will be the group at index 1 
        assert (list(grp.indices.keys()) == [0,1])
        lst = grp.indices[1]

        # create lists for consecutive indices 
        indices = []
        for _, g in groupby(enumerate(lst), lambda x: x[0] - x[1]):
            indices.append([v for _, v in g])

        lengths = [len(i) for i in indices]

        openings = [{"ix": i, "length": l} for i, l in zip(indices, lengths)]

        return openings 

    
    def calc_open_accuracy_score(self, debug=False):
        # choices and truth need to have the same indices -> ie need numerical indices, and values should be 0s or 1s 
        assert (self.choices.index == self.truth.index).all()
        # identify all the openings and determine their length 
        self.true_openings = self.determine_openings(self.truth)
        self.predicted_openings = self.determine_openings(self.choices)

        self.overlap_counter = {}
    
        scores = []
        ρ = 0.33 # de rautlin 2023, two time step limit 
        for pix, plst in enumerate(self.predicted_openings):
            self.overlap_counter [pix] = {}
            for tix, tlst in enumerate(self.true_openings):
                bool_val, len_intersect = common_member(plst["ix"], tlst["ix"])
                self.overlap_counter[pix][tix] = bool_val
                if bool_val:
                    difference = tlst['length'] - len_intersect
                    if difference == 0:
                        scores.append(1)
                    else:
                        scores.append(1 - ρ*(difference))
        
        self.open_accuracy_scores = np.array(scores)
        self.mean_open_accuracy_unbounded = np.round(np.mean(self.open_accuracy_scores),2)

        # round scores < 0 to 0 
        bound_scores = self.open_accuracy_scores.copy()
        bound_scores[bound_scores<0] = 0
        self.bound_scores = bound_scores
        self.mean_open_accuracy = np.round(np.mean(bound_scores),5)

        return self.mean_open_accuracy, self.mean_open_accuracy_unbounded

    def calc_open_instances(self):
        self.true_open_instance = 0 
        self.false_open_instance = 0

        for k, v in self.overlap_counter.items():
            if True in list(v.values()):
                self.true_open_instance +=1
                # print(k)

        self.false_open_instance = len(self.predicted_openings) - self.true_open_instance


    def calc_open_time(self):
        """choices: as opposed to guess_times, which are the times at which switches are predicted, this looks at choice(t) ~ (0, 1)
            - should be a pandas object 
        """
        self.comparison = self.truth.compare(self.choices)
        
        self.false_open_time = len(self.comparison) * self.timedelta
        self.true_open_time = (len(self.truth) - len(self.comparison)) * self.timedelta

        return self.true_open_time, self.false_open_time

    
    def calc_drdr_metrics(self,):
        self.calc_open_accuracy_score()
        self.calc_open_instances()
        self.calc_open_time()

        self.res = {
            "# true open instance": self.true_open_instance,
            "# false open instance": self.false_open_instance,
            "true open time": self.true_open_time,
            "false open time": self.false_open_time,
            "bounded acc": self.mean_open_accuracy,
            "unbounded acc": self.mean_open_accuracy_unbounded
        }

        return self.res



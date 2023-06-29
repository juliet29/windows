import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

from itertools import groupby

import sys
sys.path.insert(0, "../scripts")
import helpers as h


class Scores:
    def __init__(self, exp, near_miss_lim=2):
        """exp: a dataframe related to one of the experiments, result of h.import_desired_data  """
        self.near_miss_lim = near_miss_lim
        self.exp = exp
        self.timedelta = self.exp["DateTime"][1] - self.exp["DateTime"][0]
        self.truth = self.exp["Window Open"]

    def calc_win_change_dist(self, df, ix):
        """
        Report if the index (ix) returned by an algorithm is what the data (df) recognizes as a window flip 

        Returns
        exact: if ix reported by the algo is spot on 
        nearest: the nearest ix to what was reported 
        distance: the distance (# of indices) between the ix and the nearest found value 
        """
        # note where the value in Window Open series changes 
        shift  = df["Window Open"].shift() != df["Window Open"]

        # return if ix > lenght of data 
        if ix > len(shift):
            return "Passed index larger than it should be"
        
        # some of the experiments did not have any shifts 
        flips = np.where(shift*1==1)[0] #[1:-1]
        if len(flips) < 2:
            return "No Flip Happened!"
        
        # check if got the exact flip 
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


    def report_scores(self, guess_times, exp):
        self.hits = 0 
        self.near_miss = 0 
        self.miss = 0

        self.num_actions = len(self.exp[self.truth.shift() != self.truth])

        # see how far guess times are from actual times of window schedule change 
        self.res = pd.DataFrame(guess_times).apply(lambda x: self.calc_win_change_dist(self.exp, x.name), axis=1)

        # calculate scores and sum up 
        self.scores = self.res.apply(lambda x: self.count_score(np.abs(x[2])))
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
        self.nice_results = {k:100*np.round(v,3) for k,v in self.nice_results.items()}

        # add values that are not ratios 
        self.nice_results.update({
            "number of actions": self.num_actions,
            "number of guesses": len(self.scores)
        })

        # join the nice results and simple score sums in one dictionary 
        self.nice_results.update(self.score_sum.T.to_dict()[0])

        self.nice_res_df = pd.DataFrame.from_dict(self.nice_results, orient="index", columns=["results"])

        return  self.nice_res_df

    
    def calc_open_number(self, choices):
        """choices: as opposed to guess_times, which are the times at which switches are predicted, this looks at choice(t) ~ (0, 1)
            - should be a pandas object 
        """
        self.choices = choices 
        self.comparison = self.truth.compare(self.choices)
        
        self.false_openings = len(self.comparison)
        self.true_openings = len(self.truth) - self.false_openings

        return self.true_openings, self.false_openings

    def calc_open_time(self, choices=None):
        if choices:
            self.choices = choices

        self.true_open_time = self.truth[self.truth.astype(bool)]  * self.timedelta # need to sum up...

        self.true_close_time = self.truth[~self.truth.astype(bool)]  * self.timedelta

        self.predicted_open_times = self.choices[~self.choices.astype(bool)]  * self.timedelta

    
    
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

    
    def calc_open_accuracy_score(self, choices=None):
        if choices:
                self.choices = choices

        # choices and truth need to have the same indices 
        assert (self.choices.index == self.truth.index).all()

        # identify all the openings and determine their length 
        self.true_openings = self.determine_openings(self.truth)
        self.predicted_openings = self.determine_openings(self.choices)
    
        for pred in self.predicted_openings:
            ...
            # does pred["times"] have overlap with any of the true openings? 
            # if yes, 
                # then for how many time steps does it overlap 
                # if time steps overlap == length of opening 
                    # score = 1
                # if time steps overlap < length of opening 
                    # score = 1 - ρ*(length of opening  - length of overlap )
                

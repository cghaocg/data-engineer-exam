import random
import pandas as pd

from creme import anomaly
from creme import compose
from creme import preprocessing

class Detector:
    def __init__(self):
        self.model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(
                n_trees=20,
                height=3,
                window_size=150,
                seed=None
            )
        )
    
    def fit_predict(self, ptr):
    
        ptr = float(ptr)
        
        score = self.model.score_one({'x': ptr})
        self.model = self.model.fit_one({'x': ptr})
        
        if score > 0.8:
            pred = 1
        else:
            pred = 0

        return pred
import random
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class Detector:
    def fit_predict(self, path):
        pred = []
        vals = []
        
        df = pd.read_csv(path)

        data = df[['value']]
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # train isolation forest
        outliers_fraction = 0.01
        model =  IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        df['anomaly'] = pd.Series(model.predict(data))

        # 調整 normal=0 & anomaly=1
        df["anomaly"][df["anomaly"] == 1] = 0
        df["anomaly"][df["anomaly"] == -1] = 1
        
        for row in df.itertuples():
            vals.append(getattr(row, "value"))
            pred.append(getattr(row, "anomaly"))
        
        return vals, pred
import pandas as pd

def data_processing():
    # 讀取資料
    df_train = pd.read_csv("./data/train_need_aggregate.csv")
    df_test = pd.read_csv("./data/test_need_aggregate.csv")

    df_train["datetime"] = pd.to_datetime(df_train["datetime"])
    df_test["datetime"] = pd.to_datetime(df_test["datetime"])

    ### train
    # 依分鐘 resample
    df_train.index = df_train.datetime
    df_train = df_train.resample(rule="T").agg({"EventId": lambda x: x.tolist()}).reset_index()

    ### test
    # 依分鐘 resample
    df_test.index = df_test.datetime
    df_test = df_test.resample(rule="T").agg({"EventId": lambda x: x.tolist()}).reset_index()

    # 產 .csv
    df_train.to_csv("./result/train.csv", index=False)
    df_test.to_csv("./result/test.csv", index=False)
import pandas as pd

def data_processing():
    # 讀取資料
    df_train = pd.read_csv("./data/train_need_aggregate.csv")
    df_test = pd.read_csv("./data/test_need_aggregate.csv")

    df_train["datetime"] = pd.to_datetime(df_train["datetime"])
    df_test["datetime"] = pd.to_datetime(df_test["datetime"])


    ## train
    df_train_new = df_train.copy()
    # 依分鐘 resample
    df_train_new.index = df_train_new.datetime
    df_train_new = df_train_new.resample(rule="T").sum().reset_index()

    df_train_new["EventId"] = df_train_new["EventId"].astype(object)

    # 依分鐘放入所屬的 list
    for row in df_train_new.itertuples():
        df_train_new.at[getattr(row, "Index"), "EventId"] = (
            df_train[(df_train['datetime'] >= getattr(row, "datetime")) & 
                    (df_train['datetime'] < getattr(row, "datetime") + pd.Timedelta('1 minute'))]["EventId"].values.tolist())


    ## test
    df_test_new = df_test.copy()
    # 依分鐘 resample
    df_test_new.index = df_test_new.datetime
    df_test_new = df_test_new.resample(rule="T").sum().reset_index()

    df_test_new["EventId"] = df_test_new["EventId"].astype(object)

    # 依分鐘放入所屬的 list
    for row in df_test_new.itertuples():
        df_test_new.at[getattr(row, "Index"), "EventId"] = (
            df_test[(df_test['datetime'] >= getattr(row, "datetime")) & 
                    (df_test['datetime'] < getattr(row, "datetime") + pd.Timedelta('1 minute'))]["EventId"].values.tolist())


    # 產 .csv
    df_train_new.to_csv("./result/train.csv", index=False)
    df_test_new.to_csv("./result/test.csv", index=False) 
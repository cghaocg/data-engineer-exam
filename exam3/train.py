import os
import pandas as pd
import numpy as np
import argparse

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from model import Model

if __name__ == '__main__':
    if not os.path.exists("model"):
        os.mkdir("model")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    args = parser.parse_args()


    #================= data processing =================#
    df_train_aggr = pd.read_csv("../exam2/data/train_need_aggregate.csv")
    df_test_aggr = pd.read_csv("../exam2/data/test_need_aggregate.csv")

    df_train = pd.read_csv("../exam2/result/train.csv")
    df_test = pd.read_csv("../exam2/result/test.csv")

    # Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_train_aggr_norm = scaler.fit_transform(df_train_aggr["EventId"].values.reshape(-1, 1))

    df_train_aggr_norm = torch.FloatTensor(df_train_aggr_norm).view(-1)

    # a pattern cycle approximately
    train_window = 70

    def create_input_sequences(input_data, tw):
        input_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            input_seq.append((train_seq ,train_label))
        return input_seq

    train_input_seq = create_input_sequences(df_train_aggr_norm, train_window)


    #================= train model =================#
    input_size = 1
    hidden_layer_size = 20
    batch_size = 1
    output_size = 1
    num_layers = 1

    model = Model(input_size, hidden_layer_size, batch_size, output_size, num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    epochs = 1
    
    for i in range(epochs):
        for seq, labels in train_input_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(model.num_layers, model.batch_size, model.hidden_layer_size),
                                torch.zeros(model.num_layers, model.batch_size, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")
    
        
    #================= save model weights =================#
    model_path = "./model/lstm.pkl"
    torch.save(model.state_dict(), model_path)


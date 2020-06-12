import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size:int, hidden_layer_size:int, batch_size:int, output_size:int, num_layers:int):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        # num_layers, batch_size, self.hidden_size
        self.hidden_cell = (torch.zeros(num_layers, batch_size, hidden_layer_size),
                            torch.zeros(num_layers, batch_size, hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) , self.batch_size, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
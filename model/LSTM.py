import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Model(nn.Module) :
    def __init__(self, hidden_dim = 18, input_dim = 14 * 5 * 50, num_layers = 5, batch_size = 30) :
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )
    def forward(self, inputs):
        out ,(hidden, cell) = self.lstm(inputs, None)
        return out

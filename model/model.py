#!/usr/bin/env python3
import torch
from torch import nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, n_hiden, first_hiden = 0, center_hiden = 0 , last_hiden = 2): 
        if first_hiden == 0:
            first_hiden = input_dim
        if center_hiden == 0:
            center_hiden = first_hiden
        super(LogisticRegressionModel, self).__init__()
        self.leyer_list = nn.ModuleList()
        layer_w = input_dim
        hiden_dim = first_hiden
        for i in range(n_hiden+1):
          self.leyer_list.append(nn.Linear(layer_w, hiden_dim))
          if i < (n_hiden/2)-1:
            layer_w = hiden_dim
            hiden_dim = (int)((i+1)*(2*(center_hiden-first_hiden)/n_hiden)+first_hiden)
          else:
            layer_w = hiden_dim
            hiden_dim = (int)((i-n_hiden/2+1)*(2*(last_hiden-center_hiden)/n_hiden)+center_hiden)
        self.leyer_list.append(nn.Linear(layer_w, 1))
    def forward(self, x): 
        Relu = nn.ReLU()
        Sigm = nn.Sigmoid()
        for layer in self.leyer_list[:-1]:
            x = Relu(layer(x))
        y = Sigm(self.leyer_list[-1](x))
        return y
    def pred(self, *features):
        features = torch.tensor(features)
        return (self(features) > 0.5).item()


n_hiden = 14
input_dim = 10
first_hiden = 90
center_hiden = 70
last_hiden = 15
params = [n_hiden, first_hiden, center_hiden, last_hiden]
model = LogisticRegressionModel(input_dim, *params)
model.load_state_dict(torch.load("model{}_{}_{}_{}".format(*params), map_location=torch.device('cpu')))


if(__name__ == "__main__"):
    print(model.pred(-0.9627, -0.0090, -0.0079,  0.6588, -0.5875,  0.7196, -0.5029,  0.5987, -0.4837,  0.9176)) #False
    print(model.pred(0.6877, -0.0080, -0.0159,  0.4448, -0.5875,  0.7196, -0.3897,  0.9181, 0.1569,  0.9379)) #True
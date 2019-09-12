#!/usr/bin/env python3
import torch
import numpy as np
import pickle
from torch import nn
from sklearn.preprocessing import StandardScaler

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
        features_list = np.array([list(features)])
        features_list = self.scaler.transform(features_list)
        
        features = torch.FloatTensor(features_list)
        return (self(features[0]) > 0.5).item()
    def loadScaler(self, scaler):
        self.scaler = scaler

n_hiden = 14
input_dim = 10
first_hiden = 90
center_hiden = 70
last_hiden = 15
params = [n_hiden, first_hiden, center_hiden, last_hiden]
model = LogisticRegressionModel(input_dim, *params)
model.load_state_dict(torch.load("model{}_{}_{}_{}".format(*params), map_location=torch.device('cpu')))
with open("scaler",'rb') as filename:
    scaler = pickle.load(filename)
model.loadScaler(scaler)

if(__name__ == "__main__"):
    print(model.pred(1.447929, 1849.101146, 1083.522737, 27.0, 0.0, 12.0, -1418.736880, -143.095899, -0.722093, -0.038671)) #False
    print(model.pred(0.354945, 10085.622823, 1345.148537, 24.0, 0.0, 12.0, -393.699553, -284.193871, -0.129122, -0.037920)) #True
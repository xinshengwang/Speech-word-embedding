import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.config import cfg

class RNN(nn.Module):
    def __init__(self, input_size=80, batch_size=64, hidden_size=20, num_classes=1):
        super(RNN,self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size,hidden_size)
        #self.dropout_1 = nn.Dropout(p=0.2)
        #self.dropout_2 = nn.Dropout(p=0.2)
        self.num_layers = cfg.WBDNet.num_layers
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True,num_layers=self.num_layers,batch_first=True)
        self.fc3 = nn.Linear(hidden_size * 2,num_classes)
        self.sigm = nn.Sigmoid()

    def forward(self,x):
        out,hidden = self.lstm(x)
        out = self.fc3(out).squeeze()
        # out = self.sigm(out)
        return out
    
        
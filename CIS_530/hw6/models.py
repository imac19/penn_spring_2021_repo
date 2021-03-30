# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:02:58 2021

@author: imacd_0odruq3
"""

#models.py
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

'''
Please add default values for all the parameters of __init__.
'''
class CharRNNClassify(nn.Module):
    def __init__(self, input_size=57, hidden_size=50, output_size=9):
        super(CharRNNClassify, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.l1 = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.l2 = torch.nn.Linear(input_size + hidden_size, output_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_linear = torch.nn.Linear(hidden_size, output_size)
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True, num_layers=2, dropout=.3)
        self.gru_linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        output, hidden = self.gru(input.view(1,1,input.shape[1]), hidden)
        output = self.gru_linear(output)
        output = self.softmax(output.view(1,output.shape[2]))
        return output, hidden 

    def init_hidden(self):
        return torch.zeros(2, 1, self.hidden_size)
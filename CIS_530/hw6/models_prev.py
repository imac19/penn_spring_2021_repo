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

    def forward(self, input, hidden=None):
        combined = torch.cat((input, hidden), 1)
        hidden = self.l1(combined)
        output = self.l2(combined)
        output = self.softmax(output)
        return output, hidden 

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
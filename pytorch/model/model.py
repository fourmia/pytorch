'''
Author: lzd
Date: 2021-06-21 16:30:08
LastEditTime: 2021-06-21 16:56:38
LastEditors: Please set LastEditors
Description: ConvLSTM model structure
FilePath: \mygithub\pytorch\model\model.py
'''

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2, kernel_size[1]//2     # calculate padding range
        self.bias = bias

        self.conv = nn.Conv2d(in_channels = self.input_dim + self.hidden_dim,
                               out_channels = 4*self.hidden_dim,
                               kernel_size = self.kernel_size,
                               padding = self.padding,
                               bias = self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state                                # h_cur and x_input concat as combine_input
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.

    
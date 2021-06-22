'''
Author: lzd
Date: 2021-06-21 16:30:08
LastEditTime: 2021-06-22 10:50:23
LastEditors: Please set LastEditors
Description: ConvLSTM model structure
FilePath: \mygithub\pytorch\model\model.py
'''

import torch
import torch.nn as nn
from torch.types import Number

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
        c = torch.sigmoid(c_cur)        # according to NIPS-2015-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting-Paper 3.1

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)    # include ct-1 channles message, according fc_lstm
        i = torch.cat([torch.sigmoid(cc_i), c], dim=1)
        f = torch.cat([torch.sigmoid(cc_f), c],dim=1)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i*g

        o = torch.cat([torch.sigmoid(cc_o), c_next], dim=1)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                    batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)        # kernel_size must is  x、[x,xn]、[(x1,xn), (x2,xn),...], x is int!

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length! Makesure (kernel_size、hidden_dim and num_layers) length is same.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        self.return_all_layers = return_all_layers


        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]    # model struct:current input is last layer output, cur input dim!
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                                kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)                               # model sequence


    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permuter(1, 0, 2, 3, 4)
        b, *_, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)                # Sequence Lenth location is dim 1!
        cur_layer_input = input_tensor                #! cur_layer is 0, cur_layer_input is input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]            # initnal h,c state
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)                  # get corrent h as next layer input

            layer_output = torch.stack(output_inner, dim=1)    #! need check layer_output shape ,should is only layer out(according to seq)
            cur_layer_input = layer_output                     # next_layer input

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]     # get last output, shape is [sequence, batch, h, w] or other combination
            last_state_list = last_state_list[-1:]         # get last layer [h, c]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states



    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):             # layer extend, extended num_layer
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



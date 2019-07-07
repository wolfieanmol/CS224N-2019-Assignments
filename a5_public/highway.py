#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    '''
    creates a highway block after taking an input from conv1d block
    '''
    def __init__(self, word_embed_size, dropout_prob=0.3):
        '''
       @param  x_conv_out (batch_size, embed_size, 1)
       @param dropout_rate : probability of dropping out
        '''
        super(Highway, self).__init__()
        self.dropout_prob = dropout_prob
        # self.x_conv_out_size = list(x_conv_out.size())
        # print('####highway.py###### size of x_conv_out ', self.x_conv_out_size, x_conv_out.size())

        # self.batch_size = self.x_conv_out_size[0]
        # print(self.batch_size)
        self.word_embed_size = word_embed_size
        # print(self.word_embed_size)

        self.x_proj_layer = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.x_gate_layer = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        # print('######highway### weight', self.x_gate_layer.weight.size())

    def forward(self, x_conv_out):
        x_proj = F.relu(self.x_proj_layer(x_conv_out))
        # print('*#*# ####highway.py###### shape of x_proj', x_proj.size())

        x_gate = torch.sigmoid(self.x_gate_layer(x_conv_out))
        # print('*#*# ####highway.py###### shape of x_gate', x_proj.size())

        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1-x_gate), x_conv_out)
        # print('*#*# ####highway.py###### shape of x_highway', x_highway.size())

        x_word_embed = self.dropout_layer(x_highway)

        return x_word_embed

if __name__ == '__main__':
    x_conv_out = torch.rand((3, 2))
    highway_block = Highway(2)
    x_word_em = highway_block.forward(x_conv_out)
    # print(x_word_em.size())




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernal_size=5):
        '''
        @param x_emb (batch_size, char_embed, m_word)
        @param kernal_size (k):
        @param word_embed (f): embedding size of words in vocab. the filter size is word_embed since we want word_embed from cnn.
        '''
        super(CNN, self).__init__()

        # self.x_emb_size = list(x_emb.size())
        # print('####cnn########## x_emb_size', x_emb.size())

        # self.batch_size = self.x_emb_size[0] #8
        self.char_embed_size = char_embed_size #4
        # self.m_word = self.x_emb_size[2] #10
        self.num_filter = word_embed_size #10

        self.conv_layer = nn.Conv1d(self.char_embed_size, self.num_filter, kernel_size=kernal_size)
        self.maxpool = nn.MaxPool1d(21-kernal_size+1)

        # print('############cnn########### weight', self.conv_layer.weight.size()) #(f, e_char, k)  10, 4, 5

    def forward(self, x_emb):
        x_conv = self.conv_layer(x_emb)
        # print('##########cnn########## x_conv_size', x_conv.size() )#batch_size, e_word(f), (m_word-k+1)   8, 10, 6
        x_conv_out = self.maxpool(F.relu(x_conv))
        # print('#####cnn#############    x_conv_out size', x_conv_out.size()) #batch_size, e_word(f)    8, 10

        return x_conv_out


if __name__ == '__main__':
    x_emb = torch.rand((50,50,21))
    CNN_model = CNN(50, 3, 5)
    CNN_model.forward(x_emb)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.char_embed_size = 50
        self.dropout_prob = 0.3
        self.char_padding_index = vocab.char2id['<pad>']


        self.char_embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, self.char_padding_index)
        self.cnn = CNN(self.char_embed_size, self.embed_size, 5)
        self.highway = Highway(self.embed_size, self.dropout_prob)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        # print('###########model_embeddings######### input size ', input.size()) #(sentence_length, batch_size, m_word) 10 , 5 ,21
        x_embed = self.char_embeddings(input)                                   #(batch_size, char_embed, m_word)                5, 50, 21
        # print('###########model_embeddings######### x_embed size', x_embed.size())                                              #10, 5 ,21, 50
        x_embed_list = list(x_embed.size())
        # print('###########model_embeddings######## x_embed_list', x_embed_list)
        x_embed_reshaped = x_embed.reshape(-1, x_embed_list[3], x_embed_list[2])   #(batch_size, char_embed, m_word)  50, 50, 21 batch_size = sen_len*batch_size
        # print('###########model_embeddings######## x_embed_reshaped', x_embed_reshaped.size())  #50, 50, 21

        x_conv_out = self.cnn(x_embed_reshaped)
        # print('###########model_embeddings######### x_conv_out size', x_conv_out.size())

        x_conv_out = x_conv_out.reshape(x_embed_list[0], x_embed_list[1], -1)
        # print('###########model_embeddings######### x_conv_out size', x_conv_out.size())

        x_word_embed = self.highway(x_conv_out)
        # print('###########model_embeddings######### x_word_embedsize', x_word_embed.size())

        return x_word_embed


        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE


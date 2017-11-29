"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self,  reset_emb, batch_size, reuse, input_dim=300, num_hidden=1000):

        """

        This Class implements an Encoder.

        :param batch_size: int, length of batch
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden: int, dimensions for the hidden state of the encoder
        """

        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.reuse = reuse
        self.batch_size = batch_size

        initializer_weights = tf.variance_scaling_initializer() #xavier
        initializer_biases = tf.constant_initializer(0.0)
        self.gru_cell = tf.contrib.rnn.GRUCell(num_hidden, kernel_initializer=initializer_weights, bias_initializer=initializer_biases)
        #self.zero_state = self.gru_cell.zero_state(batch_size, tf.float32)

    def compute_state(self, x, batch_size, query_or_session_length, state=None):
        """
        :x: array of embeddings of query batch and reset vector 1 when embedding is not end of session/query 0 when it is
        :return: query representation tensor
        """
        # Initialise recurrent state
        if not state:
            state = self.gru_cell.zero_state(batch_size, tf.float32)
        # Calculate RNN states
        for step in range(query_or_session_length):
            _, state = self.gru_cell(x[step], state)

        return state
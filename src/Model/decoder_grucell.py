"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import tensorflow as tf
import numpy as np

class Decoder(object):

    def __init__(self, input_dim=300, num_hidden_query=1000, num_hidden_session=1500):
        """
        This Class implements a Decoder.
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden_query: int, dimensions for the hidden state of the decoder, same as the query_encoder
        :param num_hidden_session: int, dimensions for the hidden state of the session enconder
        """
        self.input_dim = input_dim
        self.num_hidden_query = num_hidden_query
        self.num_hidden_session = num_hidden_session

        initializer_weights = tf.variance_scaling_initializer() #xavier
        initializer_biases = tf.constant_initializer(0.0)

        self.gru_cell = tf.contrib.rnn.GRUCell(num_hidden_query, kernel_initializer=initializer_weights, bias_initializer=initializer_biases)
        # Weights for initial recurrent state
        self.Bo = tf.get_variable(shape= [self.num_hidden_query], initializer= initializer_biases, name= 'Bo')
        self.Do = tf.get_variable(shape= [self.num_hidden_query, self.num_hidden_session], initializer= initializer_biases, name= 'Do' )

    def length(sequence):
        """
         :sequence: batch of padded length; with zero vectors after eoq_symbol 
         :return:   vector determining length of queries (at what point the eoq_symbol is encountered)
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def compute_state(self, x, session_state, query_encoder_last_state):
        """
        :x:             query/session batch of padded length [batch_size x max_length x out_size]
        :session_state: state to initialise the recurrent state of the decoder
        :return:        query representation tensor [batch_size x max_length x out_size]
        """
        # Initialise recurrent state with session_state
        state = tf.tanh(tf.sum(tf.matmul(session_state, self.Do), self.Bo))
        _, state = self.gru_cell(query_encoder_last_state, state)

        # Calculate RNN states
        _, states = tf.nn.dynamic_rnn(
            self.gru_cell,
            x,
            dtype=tf.float32,
            sequence_length=self.length(x),
            initial_state=state)

        return states

    def compute_prediction(self, session_state, query_encoder_last_state, sequence_length):
        """
        :session_state:            state to initialize the recurrent state of the decoder
        :query_encoder_last_state: last encoder state of the previous query to be used as first input
        """
        outputs       = []
        state         = tf.tanh(tf.sum(tf.matmul(session_state, self.Do), self.Bo))
        output, state = self.gru_cell(query_encoder_last_state, state)
        output        = self.prediction(output)
        outputs.append(output)

        # Calculate RNN states
        # TODO: how do I check that the output is the eoq_symbol!! (or do I only run seq_length and we will see afterwards)
        stop_after = sequence_length
        while stop_after>0:
            output, state = self.gru_cell(output, state)
            output        = self.prediction(output)
            outputs.append(output)
            stop_after -= 1
        return outputs

    def check_if_eoq(self, gru_output):
        # Check if the output is the end of query symbol
        raise NotImplementedError

    def calculate_word(self, gru_output):
        # Calculate word given the output of the Decoder 
        raise NotImplementedError

    def prediction(self, gru_output):
        # Transform output to prediction given the output of the Decoder
        raise NotImplementedError

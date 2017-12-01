"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

File to implement the methods that describe each layer needed for the model.

"""

import tensorflow as tf
import numpy as np
import decoder
import encoder


def get_embedding_layer(vocabulary_size, embedding_dims, data):
    """
    Layer to train embeddings and retunr the embeddings for data
    :param vocabulary_size: int for the vocabulary size
    :param embedding_dims: int for the embedding space dimensions
    :param data: tensor with the data to obtain the embeddings from
    :return: embeddings for data

    """
    with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):

        embeddings_weights= tf.get_variable(name='embeddings', shape=[vocabulary_size, embedding_dims],
                                            initializer= tf.random_normal_initializer(mean=0.0,stddev=1.0))
        word_embeddings= tf.nn.embedding_lookup(embeddings_weights, data)

        return word_embeddings


def output_layer(embedding_dims, num_hidden, vocabulary_size, state, word):
    """
    Layer to get the output to predict the probability of the next word (word_t+1).
    The previous recurrent state of the decoder and the previous word are used.

    :param embedding_dims: int, dimensions of the embeddings vector
    :param num_hidden: int, query-level dimensionality
    :param vocabulary_size: int, vocab_size
    :param state: previous state of the encoder, should be flatten
    :param word: 1D tensor embedding, previous word, should be flatten
    :return:
    """
    with tf.variable_scope('output_layer'):
        # Define the weights H_o, E_o and bias b_o
        H_ouput = tf.get_variable(name='h_output', shape=[embedding_dims, num_hidden],
                                  initializer=tf.truncated_normal(shape=[num_hidden]))
        E_output = tf.get_variable(name='e_output', shape=[embedding_dims, vocabulary_size],
                                   initializer=tf.truncated_normal(shape=[vocabulary_size]))
        b_output = tf.get_variable(name='b_output', shape=[embedding_dims],
                                   initializer=tf.truncated_normal(shape=[embedding_dims]))

        return tf.matmul(H_ouput, state) + tf.matmul(E_output, word) + b_output

def decoder_initialise_layer(initial_session_state, hidden_dims):

    with tf.variable_scope('decoder_initial_layer', reuse=tf.AUTO_REUSE):

        return tf.contrib.layers.fully_connected(initial_session_state, hidden_dims, activation_fn= tf.nn.tanh,
                                                 weights_initializer= tf.contrib.layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer())




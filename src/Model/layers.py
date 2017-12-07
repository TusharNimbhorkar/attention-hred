"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

File to implement the methods that describe each layer needed for the model.

"""

import tensorflow as tf
import numpy as np


def get_embedding_layer(vocabulary_size, embedding_dims, data, scope):
    """
    Layer to train embeddings and retunr the embeddings for data
    :param vocabulary_size: int for the vocabulary size
    :param embedding_dims: int for the embedding space dimensions
    :param data: tensor with the data to obtain the embeddings from
    :return: embeddings for data

    """
    with tf.variable_scope(scope):
        embeddings_weights = tf.get_variable(name=scope, shape=[vocabulary_size, embedding_dims],
                                             initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        word_embeddings = tf.nn.embedding_lookup(embeddings_weights, data)

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
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        # E_output = tf.get_variable(name='e_output', shape=[embedding_dims, vocabulary_size],
        #                           initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0))
        b_output = tf.get_variable(name='b_output', shape=[embedding_dims],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

        y_one_hot = tf.one_hot(word, vocabulary_size)
        start_word = tf.expand_dims(tf.zeros([tf.shape(word)[0],vocabulary_size]),1)
        y_one_hot_shifted = tf.concat([start_word,y_one_hot],1)
        print(vocabulary_size)
        print(embedding_dims)
        print(y_one_hot_shifted)
        y_embedding_onehot = get_embedding_layer(vocabulary_size=vocabulary_size,
                                                 embedding_dims=embedding_dims, scope='output_embedding', data=y_one_hot_shifted)

        return tf.matmul(state, tf.transpose(H_ouput)) + tf.transpose(tf.cast(y_embedding_onehot, tf.float32)) + b_output


def decoder_initialise_layer(initial_session_state, hidden_dims):
    """
    Function to train the weights for the decoder init state d = tanh(D_0 * s_{m-1} + b_0

    :param initial_session_state: initial session state to initialise the decoder
    :param hidden_dims: the hidden dimensions of the RNN
    :return: returns the fully connected layer used to initialise the decoder
    """

    with tf.variable_scope('decoder_initial_layer', reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.fully_connected(initial_session_state, hidden_dims, activation_fn=tf.nn.tanh,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer())


def attention_context_layer(query_state, decoder_state, decoder_dims, encoder_dims):
    """
    Fully connected layer to calculate the attention scores (alignment model). For score(h_{n-1}, q_j), query_state is
    q_j and decoder state is h_{n-1}, so the current decoder state. Returns the attention.

    :param query_state: the query-level hidden states for the queries in the session
    :param decoder_state: current decoder output
    :return: context vector
    """
    #TODO: what should be the dimensions of attention

    w = tf.get_variable(name='weight', shape=(decoder_dims, encoder_dims),
                        initializer=tf.random_normal_initializer(stddev=0.01))

    # TODO: how would this work? How to make sure that the states are the ones for the queries in the session
    weights = tf.nn.softmax(query_state * W * decoder_state)  # TODO: use matmul
    context = tf.reduce_sum(weights * query_states)

    return context
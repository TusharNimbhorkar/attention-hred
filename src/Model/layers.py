"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

File to implement the methods that describe each layer needed for the model.

"""

import tensorflow as tf
import numpy as np


def get_embedding_layer(vocabulary_size, embedding_dims, data):
    """

    :param vocabulary_size: int for the vocabulary size
    :param embedding_dims: int for the embedding space dimensions
    :param data: tensor with the data to obtain the embeddings from
    :return: embeddings for data

    """
    with tf.variable_scope('embedding_layer'):

        embeddings_weights= tf.get_variable(name='embeddings', shape=[vocabulary_size, embedding_dims] ,
                                            initializer= tf.truncated_normal())

        word_embeddings= tf.nn.embedding_lookup(embeddings_weights, data)

        return word_embeddings
def get_query_encoder_layer():
    return
def get_session_enconder_layer():
    return
def get_decoder_layer():
    return

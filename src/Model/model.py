'''
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

'''
import tensorflow as tf
import numpy as np

class HERED():
    """"
    This Class includes the methods to build de graph for the Recurrent
      Encoder-Decoder.
    """

    def __init__(self):

        # create objects for query encoder, session encoder and decoder.
        raise NotImplementedError

    def inference(self):

        # call here tf.scan for each.

        # see if we should add an additional output layer after decoder.

        raise NotImplementedError

    def train_step(self):

        # here it would go the optimizer for the model. However, since it is now 3 RNN
        # and not all of them are optimize or might be done differently maybe this is
        # not needed anymore
        raise NotImplementedError

    def get_loss(self):
        # same as for train_step.....
        raise NotImplementedError



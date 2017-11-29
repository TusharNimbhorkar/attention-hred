"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import numpy as np


class BeamSearch(object):

    """
    Class implementing beam search. The beam search receives the new hypotheses as [batch_size, vocab_size]
    once the Tensorflow graph has been run.  The maximum number of batch_steps is handled from outside since it need to interact with Tensorflow.

    """


    def __init__(self, softmax_outputs, beam_size=5):
        """
        Initialise the BeamSearch object
        :param beam_size: size of the beam
        :param softmax_outputs: numpy of size [vocab_size], the softmax output of the last layer of the decoder

        beam_items: np.matrix of [beam_size, items?], the best hypothesis (list of word indices)
        sorted_probabilities: np.array [beam_size], probability per hypothesis

        """

        self.beam_size = beam_size  # Number of hypotheses in the beam

        self.beam_items, self.sorted_probs = self._init_beam(softmax_outputs)



    def _init_beam(self, softmax_outputs):
        """
        Adds the initial top-beam_size candidates to the beam.
        :param softmax_outputs, probabilities
        :return: beam_items, beam_size words with highest probabilities.
                 beam_probs, corresponding probabilities.
        """
        beam_items = np.expand_dims(np.transpose(np.argsort(-softmax_outputs)[:self.beam_size]), 1)
        beam_probs = np.abs(np.sort(-softmax_outputs)[:self.beam_size])

        return beam_items, beam_probs

    def beam_step(self, hypotheses):
        """
        Perform a beam_step. This receives the probabilities for the hypothesis. This would be a numpy of size[beam_size, vocab_size]
        :param hypothesis, numpy matrix of size [beam_size, vocab_size],
        :return:
        """
        # Sort probabilities and prune them
        sorted_hyp = np.argsort(-hypotheses, axis=1)[:,:self.beam_size]
        sorted_probs = np.abs(np.sort(-hypotheses, axis=1))[:,:self.beam_size]

        # Calculate total probabilities for each hypothesis (in total beam_size^2 hypotheses)
        total_probs = np.multiply(sorted_probs, np.expand_dims(np.transpose(self.sorted_probs), 1))

        # Then sort all of them to get final probabilities
        new_hypotheses = np.zeros(shape=[self.beam_size, (self.beam_items.shape[1] + 1)])
        new_probs = np.zeros(shape=[self.beam_size])
        for i in range(self.beam_size):
            # Get current maximum and its probability
            flat_ind = np.argmax(total_probs)
            new_probs[i] = np.max(total_probs)
            # Get coordinates for the maximum
            ind_r = int(flat_ind / total_probs.shape[0])
            ind_c = flat_ind - (ind_r * total_probs.shape[1])
            # Get word that corresponds to the maximum
            idx_word = sorted_hyp[ind_r, ind_c] # word that is appended to the beam hypothesis in ind_r
            # Append the word to the hypothesis indicated by ind_r
            new_hypotheses[i] = np.array([self.beam_items[ind_r], idx_word])

            # Change max to zero probability so that it is not picked again
            total_probs[ind_r, ind_c] = 0.0


        return new_hypotheses, new_probs

m = np.matrix([[38, 14, 81, 50],
       [17, 65, 60, 24],
       [64, 73, 25, 95]])



b = BeamSearch(np.asarray([0.2, 0.2, 0.3, 0.16, 0.14]), 3)
# print(b.beam_items)
# print(b.sorted_probs)

mat = np.matrix([[0.1, 0.2, 0.4, 0.06, 0.24], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.5, 0.06, 0.04, 0.3]])
b.beam_step(mat)


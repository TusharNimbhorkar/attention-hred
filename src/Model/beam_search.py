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

    def __init__(self, softmax_outputs, batch_size=None, beam_size=5):
        """
        Initialise the BeamSearch object
        :param softmax_outputs, numpy of size [vocab_size], the softmax output of the last layer of the decoder.
        :param batch_size, the batch size.
        :param beam_size, size of the beam (number of hypotheses in the beam).

        beam_items: np.matrix of [beam_size, items?], the best hypothesis (list of word indices)
        sorted_probabilities: np.array [beam_size], probability per hypothesis

        """

        self.beam_size = beam_size
        self.batch_size = batch_size # used for batch beam search
        self.beam_items, self.sorted_probs = self._init_beam_single(softmax_outputs)


    def _init_beam_single(self, softmax_outputs):
        """
        Adds the initial top-beam_size candidates to the beam for an example.
        :param softmax_outputs, probabilities
        :return: beam_items, beam_size words with highest probabilities.
                 beam_probs, corresponding probabilities.
        """
        beam_items = np.expand_dims(np.transpose(np.argsort(-softmax_outputs)[:self.beam_size]), 1)
        beam_probs = np.abs(np.sort(-softmax_outputs)[:self.beam_size])

        return beam_items, beam_probs


    def _init_beam_batch(self, softmax_outputs):
        """
        Adds the initial top-beam_size candidates to the beam for a batch.
        :param softmax_outputs, probabilities
        :return: beam_items, beam_size words with highest probabilities.
                 beam_probs, corresponding probabilities.
        """
        beam_items = np.expand_dims(np.argsort(-softmax_outputs, axis=1)[:, :self.beam_size], 2)
        beam_probs = np.abs(np.sort(-softmax_outputs, axis=1)[:, :self.beam_size])

        return beam_items, beam_probs

    def beam_step_single(self, hypotheses):
        """
        Perform a beam_step for an example. This receives the probabilities for the hypothesis. This would be a numpy of size[beam_size, vocab_size]
        :param hypotheses, numpy matrix of size [beam_size, vocab_size],
        :return:
        """
        # Sort probabilities and prune them
        sorted_hyp = np.argsort(-hypotheses, axis=1)[:, :self.beam_size]
        sorted_probs = np.abs(np.sort(-hypotheses, axis=1))[:, :self.beam_size]

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
            idx_word = sorted_hyp[ind_r, ind_c]  # word that is appended to the beam hypothesis in ind_r
            # Append the word to the hypothesis indicated by ind_r
            new_hypotheses[i] = np.array([self.beam_items[ind_r], idx_word])

            # Change max to zero probability so that it is not picked again
            total_probs[ind_r, ind_c] = 0.0

        return new_hypotheses, new_probs

    def beam_step_batch(self, hypotheses):
        """
        Perform a beam_step for a batch. This receives the probabilities for the hypothesis. This would be a numpy of size[batch_size, beam_size, vocab_size]
        :param hypotheses, numpy matrix of size [batch_size, beam_size, vocab_size],
        :return:
        """
        # Sort probabilities and prune them
        sorted_hyp = np.argsort(-hypotheses, axis=2)[:, :, :self.beam_size]
        sorted_hyp_probs = np.abs(np.sort(-hypotheses, axis=2))[:, :, :self.beam_size]

        # Calculate total probabilities for each hypothesis (in total beam_size^2 hypotheses)
        total_probs = np.einsum('ijk,ij->ijk', sorted_hyp_probs, self.sorted_probs)

        # Then sort all of them to get final probabilities
        new_hypotheses = np.zeros(shape=[self.batch_size, self.beam_size, (self.beam_items.shape[2] + 1)], dtype=np.int32)
        new_probs = np.zeros(shape=[self.batch_size, self.beam_size])
        # Do it per batch example
        for j in range(self.batch_size):
            for i in range(self.beam_size):
                # Get current maximum and its probability
                flat_ind = np.argmax(total_probs[j])
                new_probs[j,i] = np.max(total_probs[j])
                # Get coordinates for the maximum
                ind_r = int(flat_ind / total_probs[j].shape[0])
                ind_c = flat_ind - (ind_r * total_probs[j].shape[1])
                # Get word that corresponds to the maximum
                idx_word = sorted_hyp[j, ind_r, ind_c]  # word that is appended to the beam hypothesis in ind_r
                # Append the word to the hypothesis indicated by ind_r
                new_hypotheses[j, i] =  np.array([self.beam_items[j, ind_r], idx_word])
                # Change max to zero probability so that it is not picked again
                total_probs[j, ind_r, ind_c] = 0.0

        return new_hypotheses, new_probs

    def check_eoq_single(self, beam_items):
        """
        :param beam_items: numpy 2D array [beam_items, query_length] beam_items for an example.
                           Query length would be the suggestion length so far
        :return: True if end of query symbol in the beam search. False if different from eoq.
        """
        for i in range(beam_items.shape[0]):
            if beam_items[i][-1] == 1:
                return True
        return False


# -------  Code for debugging purposes - batch  ---------

# initial = np.matrix([0.2, 0.2, 0.3, 0.16, 0.14], [0.2, 0.2, 0.3, 0.16, 0.14], [0.2, 0.2, 0.3, 0.16, 0.14])
#
# mat = np.matrix([[0.1, 0.2, 0.4, 0.06, 0.24], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.5, 0.06, 0.04, 0.3],
#                  [0.2, 0.1, 0.4, 0.06, 0.24]])
#
# b = BeamSearch(mat, batch_size=4, beam_size=3)
#
# m = np.array([[[0.1, 0.2, 0.4, 0.06, 0.24], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.2, 0.4, 0.06, 0.24]],
#                  [[0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.2, 0.4, 0.06, 0.24], [0.1, 0.2, 0.4, 0.06, 0.24]],
#                  [[0.1, 0.5, 0.06, 0.04, 0.3], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.2, 0.4, 0.06, 0.24]],[[0.1, 0.5, 0.06, 0.04, 0.3], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.2, 0.4, 0.06, 0.24]]])
# b.beam_step(m)
# b.beam_step(m)

# end of query == 1

# -------  Code for debugging purposes - example  ---------

# b = BeamSearch(np.asarray([0.2, 0.2, 0.3, 0.16, 0.14]), batch_size=None, beam_size=3)
# print(b.beam_items)
# print(b.sorted_probs)
# print(b.check_eoq_single(b.beam_items))
#
# mat = np.matrix([[0.1, 0.2, 0.4, 0.06, 0.24], [0.3, 0.04, 0.1, 0.4, 0.16], [0.1, 0.5, 0.06, 0.04, 0.3]])
# b.beam_step_single(mat)

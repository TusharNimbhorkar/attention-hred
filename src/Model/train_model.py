'''
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

'''

import tensorflow as tf
import numpy as np
import _pickle as cPickle
import sys
import argparse
# Path to get batch iterator
sys.path.insert(0, '../sordoni/')
import data_iterator

# todo: put this stuff in arg.parse as well
BATCH_SIZE = 50
MAX_LENGTH = 50
N_BUCKETS = 20
MAX_STEPS = 10000000
VOCAB_SIZE = 50003
random_seed = 1234
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2

VOCAB_FILE = '../../data/input_model/train.dict.pkl'
TRAIN_FILE = '../../data/input_model/train.ses.pkl'
VALID_FILE = '../../data/input_model/valid.ses.pkl'

class Train(object):

    def __init__(self):

        self.vocab = cPickle.load(open(VOCAB_FILE, 'rb'))
        self.vocab_lookup_dict = {k: v for v, k, count in self.vocab}

        self.train_data, self.valid_data = data_iterator.get_batch_iterator(np.random.RandomState(random_seed), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': FLAGS.buckets,
            'bs': FLAGS.buckets,
            'train_session': TRAIN_FILE,
            'seqlen': FLAGS.max_length,
            'valid_session': VALID_FILE
        })

        some_variables = 0
        # ...
        #
    def train_model(self,batch_size=None):
        return
if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE,
                          help='Batch size to run trainer.')
    parser.add_argument('--max_length', type = int, default = MAX_LENGTH,
                          help='Max length.')
    parser.add_argument('--buckets', type = int, default = N_BUCKETS,
                          help='Number of buckets.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS,
                          help='Number of steps to run trainer.')
    FLAGS, unparsed = parser.parse_known_args()

    with tf.Graph().as_default():
        trainer = Train()
        trainer.train_model(batch_size=FLAGS.batch_size)

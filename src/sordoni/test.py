"""
Query suggestion hierarchical encoder-decoder code.
The code is inspired from nmt encdec code in groundhog
but we do not rely on groundhog infrastructure.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.nnet.conv3d2d import *
from collections import OrderedDict

from model import *
from utils import *
import operator

# Theano speed-up
theano.config.scan.allow_gc = False

def add_to_params(params, new_param):  #voegt toe aan params
    params.append(new_param)
    return new_param

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng  # random nummer generator
        self.parent = parent
        self.state = state
        self.__dict__.update(state)
        self.session_rec_activation = eval(self.session_rec_activation)  # gated of niet
        self.query_rec_activation = eval(self.query_rec_activation)
        self.params = []  # houdt bij wat alle parameter zijn --> tf.get_variable

class Encoder(EncoderDecoderBase):

    def init_params(self):
        """ sent weights """
        # embedding -->
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in')) # layer in de encoder
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='W_hh')) # layer die naar de hierarchical gaat --> output van encoder
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'))

        if self.query_step_type == "gated": # doen wij in de GRU -->  hier extra dingen voor nodig
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='W_hh_r'))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='W_hh_z'))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_z'))
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_r'))

        """ Context weights """ # wordt gebruikt door gated query steps
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in'))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.sdim, self.sdim)), name='Ws_hh'))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh'))

        if self.session_step_type == "gated":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_r'))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_z'))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.sdim, self.sdim)), name='Ws_hh_r'))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.sdim, self.sdim)), name='Ws_hh_z'))
            self.bs_z = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_z'))
            self.bs_r = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_r'))

    def plain_query_step(self, x_t, m_t, h_tm1, hr_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        h_t = self.query_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)
        hr_t = m_t * h_t

        return h_t, hr_t,

    def gated_query_step(self, x_t, m_t, h_tm1, hr_tm1): #m1 = van de vorige tijdstap
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = self.query_rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
        hr_t = m_t * h_t
        # return both reset state and non-reset state
        return h_t, hr_t, r_t, z_t, h_tilde # h_t is de state die je uiteindelijk wilt hebben

    def plain_session_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_update = self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_update
        return hs_t,

    def gated_session_step(self, h_t, m_t, hs_tm1):
        rs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r) + self.bs_r)
        zs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z) + self.bs_z)
        hs_tilde = self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_update = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde

        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')

        hs_t = (m_t) * hs_tm1 + (1 - m_t) * hs_update
        return hs_t, hs_tilde, rs_t, zs_t

    def approx_embedder(self, x):
        return self.W_emb[x]

    def build_encoder(self, x, xmask=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True

        # if x.ndim == 2 then
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is
        else:
            batch_size = 1

        # if it is not one_step then we initialize everything to 0
        if not one_step:
            h_0 = T.alloc(np.float32(0), batch_size, self.qdim)
            hr_0 = T.alloc(np.float32(0), batch_size, self.qdim)
            hs_0 = T.alloc(np.float32(0), batch_size, self.sdim)
        # in sampling mode (i.e. one step) we require
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs
            assert 'prev_hr' in kwargs
            assert 'prev_hs' in kwargs
            h_0 = kwargs['prev_h']
            hr_0 = kwargs['prev_hr']
            hs_0 = kwargs['prev_hs']

        xe = self.approx_embedder(x)
        if xmask == None:
            xmask = T.neq(x, self.eoq_sym)

        # Gated Encoder
        if self.query_step_type == "gated":
            f_enc = self.gated_query_step
            o_enc_info = [h_0, hr_0, None, None, None]
        else:
            f_enc = self.plain_query_step
            o_enc_info = [h_0, hr_0]

        if self.session_step_type == "gated":
            f_hier = self.gated_session_step
            o_hier_info = [hs_0, None, None, None]
        else:
            f_hier = self.plain_session_step
            o_hier_info = [hs_0]

        # Run through all the sentence (encode everything)
        if not one_step:
            _res, _ = theano.scan(
                f_enc, sequences=[xe, xmask], outputs_info=o_enc_info)
        # Make just one step further
        else:
            _res = f_enc(xe, xmask, h_0, hr_0)
        # Get the hidden state sequence
        h = _res[0]
        hr = _res[1]
        # All hierarchical sentence
        # The hs sequence is based on the original mask
        if not one_step:
            _res,  _ = theano.scan(
                f_hier, sequences=[h, xmask], outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res
        return (h, hr), hs, (_res[2], _res[3])

    def __init__(self, state, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.init_params()

class Decoder(EncoderDecoderBase):
    EVALUATION = 0
    BEAM_SEARCH = 1

    def __init__(self, state, rng, parent, encoder):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        # Take as input the encoder instance for the embeddings..
        # To modify in the future
        self.encoder = encoder
        self.trng = MRG_RandomStreams(self.seed)
        self.init_params()

    def init_params(self):
        """ Decoder weights """
        self.Wd_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='Wd_emb'))
        self.Wd_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='Wd_hh'))
        self.Wd_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in'))
        self.bd_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_hh'))
        self.Wd_s_0 = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_0'))
        self.bd_s_0 = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_s_0'))

        if self.decoder_bias_type == 'all':
            self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_q'))
        if self.query_step_type == "gated":
            self.Wd_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_r'))
            self.Wd_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_z'))
            self.Wd_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='Wd_hh_r'))
            self.Wd_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, (self.qdim, self.qdim)), name='Wd_hh_z'))
            self.bd_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_r'))
            self.bd_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_z'))
            if self.decoder_bias_type == 'all':
                self.Wd_s_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_z'))
                self.Wd_s_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_r'))
        out_target_dim = self.qdim
        if not self.maxout_out:
            out_target_dim = self.rankdim

        self.Wd_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, out_target_dim), name='Wd_out'))
        self.bd_out = add_to_params(self.params, theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='bd_out'))
        # Set up deep output
        if self.deep_out:
            self.Wd_e_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, out_target_dim), name='Wd_e_out'))
            self.bd_e_out = add_to_params(self.params, theano.shared(value=np.zeros((out_target_dim,), dtype='float32'), name='bd_e_out'))

            if self.decoder_bias_type != 'first':
                self.Wd_s_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, out_target_dim), name='Wd_s_out'))

        """ Rank """
        if hasattr(self, 'train_rank'):
            self.Wr_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, 1), name='Wr_out'))
            self.br_out = add_to_params(self.params, theano.shared(value=np.zeros((1,), dtype='float32'), name='br_out'))

    def build_rank_layer(self, hs):
        return T.dot(hs, self.Wr_out) + self.br_out

    def build_output_layer(self, hs, xd, hd):
        pre_activ = T.dot(hd, self.Wd_out)

        if self.deep_out:
            pre_activ += T.dot(xd, self.Wd_e_out) + self.bd_e_out

            if self.decoder_bias_type != 'first':
                pre_activ += T.dot(hs, self.Wd_s_out)
                # ^ if bias all, bias the deep output

        if self.maxout_out:
            pre_activ = Maxout(2)(pre_activ)

        return pre_activ

    def build_next_probs_predictor(self, hs, x, prev_hd):
        """
        Return output probabilities given prev_words x, hierarchical pass hs, and previous hd
        hs should always be the same (and should not be updated).
        """
        return self.build_decoder(hs, x, mode=Decoder.BEAM_SEARCH, prev_hd=prev_hd)

    def approx_embedder(self, x):
        # Here we use the same embeddings learnt in the encoder !!!
        return self.encoder.approx_embedder(x)

    def output_softmax(self, pre_activ):
        # returns a (timestep, bs, idim) matrix (huge)
        return SoftMax(T.dot(pre_activ, self.Wd_emb.T) + self.bd_out)

    def build_decoder(self, hs, x, xmask=None, y=None, y_neg=None, mode=EVALUATION, prev_hd=None, step_num=None):
        # Check parameter consistency
        if mode == Decoder.EVALUATION:
            assert not prev_hd
            assert y
        else:
            assert not y
            assert prev_hd

        # if mode == EVALUATION
        #   xd = (timesteps, batch_size, qdim)
        #
        # if mode != EVALUATION
        #   xd = (n_samples, dim)
        xd = self.approx_embedder(x)
        if not xmask:
            xmask = T.neq(x, self.eoq_sym)

        # we must zero out the </s> embedding
        # i.e. the embedding x_{-1} is the 0 vector
        # as well as hd_{-1} which will be reseted in the scan functions
        if xd.ndim != 3:
            assert mode != Decoder.EVALUATION # So only in beam search
            xd = (xd.dimshuffle((1, 0)) * xmask).dimshuffle((1, 0))
        else:
            assert mode == Decoder.EVALUATION # So only in beam search
            xd = (xd.dimshuffle((2,0,1)) * xmask).dimshuffle((1,2,0))

        # Run the decoder
        if mode == Decoder.EVALUATION:
            hd_init = T.alloc(np.float32(0), x.shape[1], self.qdim)
        else:
            hd_init = prev_hd

        if self.query_step_type == "gated":
            f_dec = self.gated_step
            o_dec_info = [hd_init, None, None, None]
        else:
            f_dec = self.plain_step
            o_dec_info = [hd_init]

        # If the mode of the decoder is EVALUATION
        # then we evaluate by default all the sentence
        # xd - i.e. xd.ndim == 3, xd = (timesteps, batch_size, qdim)
        if mode == Decoder.EVALUATION:
            _res, _ = theano.scan(f_dec,
                              sequences=[xd, xmask, hs],\
                              outputs_info=o_dec_info)
        # else we evaluate only one step of the recurrence using the
        # previous hidden states and the previous computed hierarchical
        # states.
        else:
            _res = f_dec(xd, xmask, hs, prev_hd)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hd = _res[0]
        else:
            hd = _res

        pre_activ = self.build_output_layer(hs, xd, hd)

        # EVALUATION  : Return target_probs + all the predicted ranks
        # target_probs.ndim == 3
        if mode == Decoder.EVALUATION:
            target_probs = GrabProbs(self.output_softmax(pre_activ), y)
            return target_probs, hd, _res
        # BEAM_SEARCH : Return output (the softmax layer) + the new hidden states
        elif mode == Decoder.BEAM_SEARCH:
            return self.output_softmax(pre_activ), hd

    def gated_step(self, xd_t, m_t, hs_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        hd_tm1 = (m_t) * hd_tm1 + (1 - m_t) * self.query_rec_activation(T.dot(hs_t, self.Wd_s_0) + self.bd_s_0)
        # hd_{t - 1} = tanh(W_s_0 hs_t + bd_s_0) else hd_{t - 1} is left unchanged (m_t = 1)

        # In the 'all' decoder bias type each hidden state of the decoder
        # RNN receives the hs_t vector as bias without modification
        if self.decoder_bias_type == 'all':
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + T.dot(hs_t, self.Wd_s_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + T.dot(hs_t, self.Wd_s_z) + self.bd_z)
            hd_tilde = self.query_rec_activation(T.dot(xd_t, self.Wd_in)
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh)
                                        + T.dot(hs_t, self.Wd_s_q)
                                        + self.bd_hh)
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde
            output = (hd_t, rd_t, zd_t, hd_tilde)
        else:
            # Do not bias all the decoder (force to store very useful information in the first state)
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + self.bd_r)
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + self.bd_z)
            hd_tilde = self.query_rec_activation(T.dot(xd_t, self.Wd_in)
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh)
                                        + self.bd_hh)
            hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde
            output = (hd_t, rd_t, zd_t, hd_tilde)
        return output

    def plain_step(self, xd_t, m_t, hs_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        # We already assume that xd are zeroed out
        hd_tm1 = (m_t) * hd_tm1 + (1 - m_t) * self.query_rec_activation(T.dot(hs_t, self.Wd_s_0) + self.bd_s_0)
        # ^ iff x_{t - 1} = </s> (m_t = 0) then x_{t-1} = 0
        # and hd_{t - 1} = 0 else hd_{t - 1} is left unchanged (m_t = 1)

        if self.decoder_bias_type == 'first':
            # Do not bias all the decoder (force to store very useful information in the first state)
            hd_t = self.query_rec_activation( T.dot(xd_t, self.Wd_in)
                                             + T.dot(hd_tm1, self.Wd_hh)
                                             + self.bd_hh )
            output = (hd_t,)
        elif self.decoder_bias_type == 'all':
            hd_t = self.query_rec_activation( T.dot(xd_t, self.Wd_in)
                                             + T.dot(hd_tm1, self.Wd_hh)
                                             + T.dot(hs_t, self.Wd_s_q)
                                             + self.bd_hh )
            output = (hd_t,)
        return output
    ####

class SessionEncoderDecoder(Model):
    def indices_to_words(self, seq, exclude_start_end=False):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        def convert():
            for word_index in seq:
                if word_index > len(self.idx_to_str):
                    raise ValueError('Word index is too large for the model vocabulary!')
                if word_index == self.eos_sym:
                    break
                if not exclude_start_end or (word_index != self.eoq_sym and word_index != self.soq_sym):
                    yield self.idx_to_str[word_index]
        return list(convert())

    def words_to_indices(self, seq):
        """
        Converts a list of words to a list
        of word ids. Use unk_sym if a word is not
        known.
        """
        return [self.str_to_idx.get(word, self.unk_sym) for word in seq]

    def compute_updates(self, training_cost, params):
        updates = []
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))
        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []

        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))

        grads = OrderedDict(clip_grads)

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!")
        return updates

    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")
            self.train_fn = theano.function(
                inputs=[self.x_data, self.x_ranks, self.x_max_length, self.x_cost_mask],
                outputs=self.training_cost, updates=self.updates, name="train_fn")
        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data, self.x_ranks, self.x_max_length, self.x_cost_mask],
                                           outputs=self.training_cost, name="eval_fn")
        return self.eval_fn

    def build_score_function(self):
        if not hasattr(self, 'score_fn'):
            self.score_fn = theano.function(
                inputs=[self.x_data, self.x_max_length],
                outputs=[self.per_example_cost],
                name="score_fn")
        return self.score_fn

    def build_rank_prediction_function(self):
        if not hasattr(self, 'rank_fn'):
            (h, hr), hs, _ = self.encoder.build_encoder(self.aug_x_data)
            ranks = self.decoder.build_rank_layer(hs)
            self.rank_fn = theano.function(
                inputs=[self.x_data],
                outputs=[ranks],
                name="rank_fn")
        return self.rank_fn

    def build_get_states_function(self):
        if not hasattr(self, 'get_states_fn'):
            # Compile functions
            logger.debug("Get states of the network")
            outputs = [self.h, self.hs, self.hd, self.rs, self.us] + [x for x in self.decoder_states]
            self.get_states_fn = theano.function(inputs=[self.x_data, self.x_max_length],
                                            outputs=outputs, name="get_states_fn")
        return self.get_states_fn

    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):
            outputs, hd = self.decoder.build_next_probs_predictor(
                self.beam_hs, self.beam_source, prev_hd=self.beam_hd)
            self.next_probs_fn = theano.function(
                inputs=[self.beam_hs, self.beam_source, self.beam_hd],
                outputs=[outputs, hd],
                name="next_probs_fn")
        return self.next_probs_fn

    def build_first_vector(self):
        if not hasattr(self, 'first_vec_fn'):
            (h, hr), hs, _ = self.encoder.build_encoder(self.aug_x_data)
            hd0 = self.decoder.query_rec_activation(T.dot(hs, self.decoder.Wd_s_0) + self.decoder.bd_s_0)
            self.first_vec_fn = theano.function(inputs=[self.x_data],
                outputs=[h, hs, hd0], name="first_vec_fn")
        return self.first_vec_fn

    def build_encoder_function(self):
        if not hasattr(self, 'encoder_fn'):
            (h, hr), hs, _ = self.encoder.build_encoder(self.aug_x_data)
            self.encoder_fn = theano.function(inputs=[self.x_data],
                outputs=[h, hr, hs], name="encoder_fn")
        return self.encoder_fn

    def __init__(self, state):
        Model.__init__(self)
        self.state = state

        # Compatibility towards older models
        self.__dict__.update(state)
        self.rng = numpy.random.RandomState(state['seed'])

        # Load dictionary
        raw_dict = cPickle.load(open(self.dictionary, 'r'))

        # Probabilities for each term in the corpus
        self.noise_probs = [x[2] for x in sorted(raw_dict, key=operator.itemgetter(1))]
        self.noise_probs = numpy.array(self.noise_probs, dtype='float64')
        self.noise_probs /= numpy.sum(self.noise_probs)
        self.noise_probs = self.noise_probs ** 0.75
        self.noise_probs /= numpy.sum(self.noise_probs)

        self.t_noise_probs = theano.shared(self.noise_probs.astype('float32'), 't_noise_probs')
        # Dictionaries to convert str to idx and vice-versa
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq in raw_dict])

        if '</q>' not in self.str_to_idx \
           or '</s>' not in self.str_to_idx:
                raise Exception("Error, malformed dictionary!")

        # Number of words in the dictionary
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim

        logger.debug("Initializing encoder")
        self.encoder = Encoder(self.state, self.rng, self)
        logger.debug("Initializing decoder")
        self.decoder = Decoder(self.state, self.rng, self, self.encoder)

        # Init params
        self.params = self.encoder.params + self.decoder.params
        assert len(set(self.params)) == (len(self.encoder.params) + len(self.decoder.params))

        self.y_neg = T.itensor3('y_neg')
        self.x_data = T.imatrix('x_data')
        self.x_ranks = T.imatrix('x_ranks')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_max_length = T.iscalar('x_max_length')

        # The training is done with a trick. We append a special </q> at the beginning of the dialog
        # so that we can predict also the first sent in the dialog starting from the dialog beginning token (</q>).
        self.aug_x_data = T.concatenate([T.alloc(np.int32(self.eoq_sym), 1, self.x_data.shape[1]), self.x_data])
        training_x = self.aug_x_data[:self.x_max_length]
        training_y = self.aug_x_data[1:self.x_max_length+1]
        training_ranks = self.x_ranks[:self.x_max_length-1].flatten()
        training_ranks_mask = T.neq(training_ranks, 0).flatten()
        # Here we find the end-of-sentence tokens in the minibatch.
        training_hs_mask = T.neq(training_x, self.eoq_sym)
        training_x_cost_mask = self.x_cost_mask[:self.x_max_length].flatten()

        # Backward compatibility
        if 'decoder_bias_type' in self.state:
            logger.debug("Decoder bias type {}".format(self.decoder_bias_type))

        logger.info("Build encoder")
        (self.h, _), self.hs, (self.rs, self.us) = \
          self.encoder.build_encoder(training_x, xmask=training_hs_mask)

        logger.info("Build decoder (EVAL)")
        target_probs, self.hd, self.decoder_states = \
            self.decoder.build_decoder(self.hs, training_x, xmask=training_hs_mask, \
                                                                                y=training_y, mode=Decoder.EVALUATION)
        logger.info("Build rank predictor")
        self.predicted_ranks = self.decoder.build_rank_layer(self.hs)

        # Prediction cost and rank cost
        self.per_example_cost = -T.log2(target_probs).reshape((self.x_max_length, self.x_data.shape[1]))
        self.rank_cost = T.sum(((self.predicted_ranks[1:].flatten() - training_ranks) ** 2) * (training_ranks_mask)) / T.sum(training_ranks_mask)
        self.training_cost = T.sum(-T.log2(target_probs) * training_x_cost_mask) + np.float32(self.lambda_rank) * self.rank_cost
        self.updates = self.compute_updates(self.training_cost / training_x.shape[1], self.params)

        # Beam-search variables
        self.beam_source = T.lvector("beam_source")
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")

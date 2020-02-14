"""
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import time
import math
import os
import sys
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

try:
    from transformer_model import TransformerLM
except ImportError:
    from .transformer_model import TransformerLM

try:
    from transformer_cache import CacheCell
except ImportError:
    from .transformer_cache import CacheCell

import warnings
import numpy as np
from numpy.testing import assert_almost_equal

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))


parser = argparse.ArgumentParser(description=
                                 'MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='full',
                    help='model (full, cascade, dilated, dilated_mem)')
parser.add_argument('--data', type=str, default='wikitext2',
                    help='dataset (wikitext2, wikitext103, ptb)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--char_emsize', type=int, default=200,
                    help='size of char embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--warmup_steps', type=float, default=4000,
                    help='number of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=180,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.2,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--lr_update_interval', type=int, default=30,
                    help='lr udpate interval')
parser.add_argument('--lr_update_factor', type=float, default=0.1,
                    help='lr udpate factor')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wd', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--test_mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--num_heads', type=int, default=8, metavar='N',
                    help='number of heads in multi-head attention')
parser.add_argument('--scaled', action='store_true',
                    help='whether to apply scale dot')
parser.add_argument('--units', type=int, default=512, metavar='N',
                    help='number of units used in attention layers?')
parser.add_argument('--use_residual', action='store_true',
                    help='whether to use residual connection')
parser.add_argument('--max_src_length', type=int, default=8, metavar='N',
                    help='maximum source length')
parser.add_argument('--weight_initializer', type=str, default=None, metavar='N',
                    help='weight_initializer')
parser.add_argument('--bias_initializer', type=str, default='zeros', metavar='N',
                    help='bias_initializer')
parser.add_argument('--ntasgd', action='store_true',
                    help='Whether to apply ntasgd')
parser.add_argument('--ema', action='store_true',
                    help='Whether to apply ema')
parser.add_argument('--label_smoothing', action='store_true',
                    help='Whether to apply label_smoothing')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='epsilon parameter for label smoothing')
parser.add_argument('--first_window_size', type=float, default=2,
                    help='the smallest window size')
parser.add_argument('--window_size_multiplier', type=float, default=2,
                    help='the times of window size become larger layer after layer')
parser.add_argument('--use_weight_dropout', action='store_true',
                    help='Whether to apply weight_dropout on the encoder')
parser.add_argument('--use_encoder_last_variational_dropout', action='store_true',
                    help='Whether to apply variational dropout')
parser.add_argument('--use_encoder_variational_dropout', action='store_true',
                    help='Whether to apply variational dropout')
parser.add_argument('--use_encoder_cell_variational_dropout', action='store_true',
                    help='Whether to apply variational dropout')
parser.add_argument('--use_encoder_ffn_variational_dropout', action='store_true',
                    help='Whether to apply variational dropout')
parser.add_argument('--use_layer_norm', action='store_true',
                    help='Whether to apply layer norm')
parser.add_argument('--use_pretrained_embedding', action='store_true',
                    help='whether to use pretrained embedding')
parser.add_argument('--use_highway', action='store_true',
                    help='whether to use highway')
parser.add_argument('--use_cnn_embedding', action='store_true',
                    help='whether to use cnn embedding')
parser.add_argument('--max_word_length', type=int, default=None, metavar='N',
                    help='number of units used in attention layers')
parser.add_argument('--use_cache', action='store_true',
                    help='Whether to use cache')
parser.add_argument('--cache_bptt', type=int, default=2000,
                    help='sequence length')
parser.add_argument('--window', type=int, default=2000,
                    help='cache window length')
parser.add_argument('--theta', type=float, default=0.662,
                    help='the scala controls the flatness of the cache distribution '
                         'that predict the next word')
parser.add_argument('--lambdas', type=float, default=0.1279,
                    help='linear scalar between only cache and vocab distribution')
parser.add_argument('--path_to_params_file', type=str, default=None,
                    help='path to the saved params file of user pre-trained model, '
                         'including the params file, e.g., ~/.mxnet/models/awd_lstm_lm_1150.params')
parser.add_argument('--cache_output', action='store_true',
                    help='Whether to cache output')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size in dilated attention')
parser.add_argument('--d_base', type=int, default=2,
                    help='dilation factor base')
parser.add_argument('--continue_train', action='store_true',
                    help='Whether to continue the previous training, args.save must be specified')

args = parser.parse_args()

###############################################################################
# Load data
###############################################################################


def test_batchify():
    batch_size = 2
    max_word_length = 5
    text_data = "abcd efg abc ef ab e"
    text_data_tokens = text_data.split(' ')
    counter = nlp.data.count_tokens(text_data_tokens)
    vocab = nlp.Vocab(counter, unknown_token=None, reserved_tokens=None)

    char_counter = nlp.data.count_tokens(list(text_data))
    char_vocab = nlp.Vocab(char_counter, unknown_token=None, reserved_tokens=None)

    sample_len = len(text_data_tokens) // batch_size
    word_nd = mx.nd.array(
        vocab[text_data_tokens[:sample_len * batch_size]]).reshape(
        batch_size, -1).T
    char_nd = mx.nd.array(
        [char_vocab[c] for token in text_data_tokens[:sample_len * batch_size] for c in
         list(token)[:max_word_length]])

    char_array = []
    for token in text_data_tokens[:sample_len * batch_size]:
        if len(token) < max_word_length:
            for c in list(token):
                char_array.append(char_vocab[c])
            for i in range(max_word_length - len(token)):
                char_array.append(0)
        else:
            for c in list(token)[:max_word_length]:
                char_array.append(char_vocab[c])
    char_new_nd = mx.nd.array(char_array).reshape(sample_len * batch_size, -1)

    char_nd_batches = char_new_nd.split(axis=0, num_outputs=batch_size)
    char_batches_reshaped = []
    for i in range(len(char_nd_batches[0])):
        char_batch = []
        for j in range(len(char_nd_batches)):
            char_batch.append(char_nd_batches[j][i])
        char_nd_batch = mx.nd.concat(*char_batch, dim=0).reshape(-1, max_word_length)
        char_batches_reshaped.append(char_nd_batch)
    char_nd_batches_reshaped = mx.nd.concat(*char_batches_reshaped, dim=0)\
        .reshape(sample_len, batch_size, max_word_length)


context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

assert args.weight_dropout > 0 or (args.weight_dropout == 0 and args.alpha == 0), \
    'The alpha L2 regularization cannot be used with standard RNN, please set alpha to 0'

if args.data == 'wikitext2':
    train_dataset, val_dataset, test_dataset = \
        [nlp.data.WikiText2(segment=segment,
                            skip_empty=False, bos=None, eos='<eos>')
         for segment in ['train', 'val', 'test']]
elif args.data == 'wikitext103':
    train_dataset, val_dataset, test_dataset = \
        [nlp.data.WikiText103(segment=segment,
                            skip_empty=False, bos=None, eos='<eos>')
         for segment in ['train', 'val', 'test']]
elif args.data == 'ptb':
    train_dataset = nlp.data.CorpusDataset('ptb.train.txt', flatten=True, skip_empty=False,
                                           bos=None, eos='<eos>', tokenizer=lambda s: s.split())
    val_dataset = nlp.data.CorpusDataset('ptb.valid.txt', flatten=True, skip_empty=False,
                                           bos=None, eos='<eos>', tokenizer=lambda s: s.split())
    test_dataset = nlp.data.CorpusDataset('ptb.test.txt', flatten=True, skip_empty=False,
                                           bos=None, eos='<eos>', tokenizer=lambda s: s.split())



vocab = nlp.Vocab(counter=nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

char_vocab = None
if args.use_cnn_embedding:
    def char_splitter(s):
        """Split a string at whitespace.

        Parameters
        ----------
        s : str
            The string to be split

        Returns
        --------
        List[str]
            List of strings. Obtained by calling s.split().
        """
        return list(s)

    if args.data == 'wikitext2':
        train_char_dataset = nlp.data.WikiText2(segment='train', skip_empty=False,
                                                bos=None, eos='<eos>', tokenizer=char_splitter)
        char_vocab = nlp.Vocab(counter=nlp.data.Counter(train_char_dataset), padding_token=None, bos_token=None)
    elif args.data == 'wikitext103':
        train_char_dataset = nlp.data.WikiText103(segment='train', skip_empty=False,
                                                bos=None, eos='<eos>', tokenizer=char_splitter)
        char_vocab = nlp.Vocab(counter=nlp.data.Counter(train_char_dataset), padding_token=None,
                               bos_token=None)
    elif args.data == 'ptb':
        train_char_dataset = nlp.data.CorpusDataset('ptb.train.txt', flatten=True, skip_empty=False,
                                               bos=None, eos='<eos>', tokenizer=char_splitte)
        char_vocab = nlp.Vocab(counter=nlp.data.Counter(train_char_dataset), padding_token=None,
                               bos_token=None)


class CorpusBatchify(object):
    """Transform the dataset into N independent sequences, where N is the batch size.

    Parameters
    ----------
    vocab : gluonnlp.Vocab
        The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
        index according to the vocabulary.
    batch_size : int
        The number of samples in each batch.
    """

    def __init__(self, vocab, batch_size, char_vocab=None, max_word_length=None):
        self._vocab = vocab
        self._batch_size = batch_size
        self._char_vocab = char_vocab
        self._max_word_length = max_word_length

    def __call__(self, data):
        """Batchify a dataset.

        Parameters
        ----------
        data : mxnet.gluon.data.Dataset
            A flat dataset to be batchified.

        Returns
        -------
        mxnet.gluon.data.Dataset
            NDArray of shape (len(data) // N, N) where N is the batch_size
            wrapped by a mxnet.gluon.data.SimpleDataset. Excessive tokens that
            don't align along the batches are discarded.
        """
        sample_len = len(data) // self._batch_size
        word_nd = mx.nd.array(
            self._vocab[data[:sample_len * self._batch_size]]).reshape(
            self._batch_size, -1).T
        if char_vocab is None:
            return word_nd, None
        char_array = []
        for token in data[:sample_len * self._batch_size]:
            if len(token) < self._max_word_length:
                for c in list(token):
                    char_array.append(self._char_vocab[c])
                for i in range(self._max_word_length - len(token)):
                    char_array.append(0)
            else:
                for c in list(token)[:self._max_word_length]:
                    char_array.append(self._char_vocab[c])
        char_nd = mx.nd.array(char_array).reshape(sample_len * self._batch_size, -1)
        char_nd_batches = char_nd.split(axis=0, num_outputs=self._batch_size)
        char_batches_reshaped = []
        for i in range(len(char_nd_batches[0])):
            char_batch = []
            for j in range(len(char_nd_batches)):
                char_batch.append(char_nd_batches[j][i])
            char_nd_batch = mx.nd.concat(*char_batch, dim=0).reshape(-1, self._max_word_length)
            char_batches_reshaped.append(char_nd_batch)
        char_nd_batches_reshaped = mx.nd.concat(*char_batches_reshaped, dim=0) \
            .reshape(sample_len, self._batch_size, self._max_word_length)
        char_nd = char_nd_batches_reshaped
        return word_nd, char_nd



train_batchify = CorpusBatchify(vocab, char_vocab=char_vocab, batch_size=args.batch_size, max_word_length=args.max_word_length)
train_data, train_char_data = train_batchify(train_dataset)

val_batch_size = args.batch_size
val_batchify = CorpusBatchify(vocab, char_vocab=char_vocab, batch_size=val_batch_size, max_word_length=args.max_word_length)
val_data, val_char_data = val_batchify(val_dataset)

test_batch_size = args.batch_size
test_batchify = CorpusBatchify(vocab, char_vocab=char_vocab, batch_size=test_batch_size, max_word_length=args.max_word_length)
test_data, test_char_data = test_batchify(test_dataset)

#TODO: if sample equal to the size to wikitext103
# train_data = train_data[104431:208862]

if args.test_mode:
    args.emsize = 300
    args.nhid = 400
    args.units = 300
    args.num_heads = 2
    args.nlayers = 2
    args.epochs = 3
    args.log_interval = 1
    args.cache_bptt = 99
    args.window = 99
    train_data = train_data[0:100]
    val_data = val_data[0:100]
    test_data = test_data[0:100]

#TODO: better change to train, val, test, since the valid length of the last batch might be different
#TODO: but this simple hack doesn't impact the final results.
def get_data_lengths(ctx):
    lengths = []
    for i in range(args.batch_size):
        lengths.append(args.bptt)
    return mx.nd.array(lengths).as_in_context(ctx)

valid_lengths = get_data_lengths(context[0])

###############################################################################
# Build the model
###############################################################################

ntokens = len(vocab)

#TODO: apply weight drop to the weight matrix in the attention, dropout_h??
model = TransformerLM(model=args.model,
                      vocab_size=len(vocab),
                      char_vocab_size=len(char_vocab) if char_vocab is not None else 0,
                      embed_size=args.emsize,
                      char_embed_size=args.char_emsize,
                      hidden_size=args.nhid,
                      num_layers=args.nlayers,
                      tie_weights=args.tied,
                      dropout=args.dropout,
                      weight_drop=args.weight_dropout,
                      drop_h=args.dropout_h,
                      drop_i=args.dropout_i,
                      drop_e=args.dropout_e,
                      num_heads=args.num_heads,
                      scaled=args.scaled,
                      units=args.units,
                      use_residual=args.use_residual,
                      max_src_length=args.max_src_length,
                      window_size=args.first_window_size,
                      window_size_multiplier=args.window_size_multiplier,
                      use_weight_dropout=args.use_weight_dropout,
                      use_encoder_last_variational_dropout=args.use_encoder_last_variational_dropout,
                      use_encoder_variational_dropout=args.use_encoder_variational_dropout,
                      use_encoder_cell_variational_dropout=args.use_encoder_cell_variational_dropout,
                      use_encoder_ffn_variational_dropout=args.use_encoder_ffn_variational_dropout,
                      use_layer_norm=args.use_layer_norm,
                      use_pretrained_embedding=args.use_pretrained_embedding,
                      use_highway=args.use_highway,
                      use_cnn_embedding=args.use_cnn_embedding,
                      kernel_size=args.kernel_size,
                      d_base=args.d_base,
                      weight_initializer=args.weight_initializer,
                      bias_initializer=args.bias_initializer)


if args.continue_train:
    model.load_parameters(args.save, ctx=context)
else:
    model.initialize(mx.init.Xavier(), ctx=context)

static_alloc = True
if not args.test_mode:
    model.hybridize(static_alloc=static_alloc)

if args.use_pretrained_embedding and not args.continue_train:
    glove_embedding = nlp.embedding.create('glove', source='glove.840B.300d')
    vocab.set_embedding(glove_embedding)
    model.embedding.word_emb[0].weight.set_data(vocab.embedding.idx_to_vec.as_in_context(context[0]))
    assert_almost_equal(vocab.embedding.idx_to_vec[vocab.embedding.token_to_idx['his']].asnumpy(),
                        glove_embedding.idx_to_vec[glove_embedding.token_to_idx['his']].asnumpy(),
                        decimal=4)

    assert_almost_equal(vocab.embedding.idx_to_vec[vocab.embedding.token_to_idx['one']].asnumpy(),
                        glove_embedding.idx_to_vec[glove_embedding.token_to_idx['one']].asnumpy(),
                        decimal=4)

if args.optimizer == 'sgd':
    if args.use_pretrained_embedding:
        trainer_params = {'learning_rate': args.lr,
                          'momentum': 0}
    else:
        trainer_params = {'learning_rate': args.lr,
                          'momentum': 0,
                          'wd': args.wd}
elif args.optimizer == 'adam':
    if args.use_pretrained_embedding:
        trainer_params = {'learning_rate': args.lr,
                          'beta1': 0,
                          'beta2': 0.999,
                          'epsilon': 1e-9}
    else:
        trainer_params = {'learning_rate': args.lr,
                          'wd': args.wd,
                          'beta1': 0,
                          'beta2': 0.999,
                          'epsilon': 1e-9}

trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params, update_on_kvstore=False)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
if not args.test_mode:
    loss.hybridize(static_alloc=static_alloc)

class ExponentialMovingAverage():
    r"""An implement of Exponential Moving Average.
         shadow variable = decay * shadow variable + (1 - decay) * variable
     Parameters
    ----------
    decay : float, default 0.9999
        The axis to sum over when computing softmax and entropy.
    """
    def __init__(self, decay=0.9999, **kwargs):
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self.decay = decay
        self.shadow = {}

    def add(self, name, parameters):
        r"""Update the shadow variable.
         Parameters
        -----------
        name : string
            the name of shadow variable.
        parameters : NDArray
            the init value of shadow variable.
        Returns
        --------
        return : None
        """
        self.shadow[name] = parameters.copy()

    def __call__(self, name, x):
        r"""Update the shadow variable.
         Parameters
        -----------
        name : string
            the name of shadow variable.
        x : NDArray
            the value of shadow variable.
        Returns
        --------
        return : None
        """
        assert name in self.shadow
        self.shadow[name] = self.decay * \
            self.shadow[name] + (1.0 - self.decay) * x

    def get(self, name):
        r"""Return the shadow variable.
         Parameters
        -----------
        name : string
            the name of shadow variable.
         Returns
        --------
        return : NDArray
            the value of shadow variable.
        """
        return self.shadow[name]

ema_decay = 0.9999
if args.ema:
    print('Use EMA!')
    ema = ExponentialMovingAverage(decay=ema_decay)
else:
    ema = None


class LabelSmoothing(gluon.HybridBlock):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Parameters
    ----------
    axis : int, default -1
        The axis to smooth.
    epsilon : float, default 0.1
        The epsilon parameter in label smoothing
    sparse_label : bool, default True
        Whether input is an integer array instead of one hot array.
    units : int or None
        Vocabulary size. If units is not given, it will be inferred from the input.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, axis=-1, epsilon=0.1, units=None,
                 sparse_label=False, prefix=None, params=None):
        super(LabelSmoothing, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._epsilon = epsilon
        self._sparse_label = sparse_label
        self._units = units

    def hybrid_forward(self, F, inputs, units=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        inputs : Symbol or NDArray
            Shape (batch_size, length) or (batch_size, length, V)
        units : int or None
        Returns
        -------
        smoothed_label : Symbol or NDArray
            Shape (batch_size, length, V)
        """
        if self._sparse_label:
            assert units is not None or self._units is not None, \
                'units needs to be given in function call or ' \
                'instance initialization when sparse_label is False'
            if units is None:
                units = self._units
            inputs = F.one_hot(inputs, depth=units)
        if units is None and self._units is None:
            return F.Custom(inputs, epsilon=self._epsilon, axis=self._axis,
                            op_type='_smoothing_with_dim')
        else:
            if units is None:
                units = self._units
            return ((1 - self._epsilon) * inputs) + (self._epsilon / units)


if args.label_smoothing:
    print('Use label smoothing')
    label_smoothing = LabelSmoothing(epsilon=args.epsilon, units=len(vocab))
    if not args.test_mode:
        label_smoothing.hybridize(static_alloc=static_alloc)


###############################################################################
# Training code
###############################################################################

def detach(hidden):
    """Transfer hidden states into new states, to detach them from the history.
    Parameters
    ----------
    hidden : NDArray
        The hidden states
    Returns
    ----------
    hidden: NDArray
        The detached hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def get_batch(data_source, char_data_source, i, seq_len=None):
    """Get mini-batches of the dataset.

    Parameters
    ----------
    data_source : NDArray or mxnet.gluon.Dataset
        The dataset is evaluated on.
    i : int
        The index of the batch, starting from 0.
    seq_len : int
        The length of each sample in the batch.

    Returns
    -------
    data: NDArray
        The context
    target: NDArray
        The words to predict
    """
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    if char_data_source is None:
        return data, target, None, None
    char_data = char_data_source[i:i+seq_len]
    char_target = char_data_source[i+1:i+1+seq_len]
    return data, target, char_data, char_target



def evaluate(data_source, char_data_source, batch_size, params_file_name, ctx=None, ema=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_source : NDArray
        The dataset is evaluated on.
    batch_size : int
        The size of the mini-batch.
    params_file_name : str
        The parameter file to use to evaluate,
        e.g., val.params or args.save
    ctx : mx.cpu() or mx.gpu()
        The context of the computation.

    Returns
    -------
    loss: float
        The loss on the dataset
    """

    total_L = 0.0
    ntotal = 0

    model.load_parameters(params_file_name, context)
    if ema is not None:
        for name, params in model.collect_params().items():
            params.set_data(ema.get(name))

    # if data_source == val_data:
    #     valid_lengths = val_valid_lengths
    # elif data_source == test_data:
    #     valid_lengths = test_valid_lengths
    i = 0
    masks = None
    while i < len(data_source) - 1 - 1:
        data, target, char_data, _ = get_batch(data_source, char_data_source, i, seq_len=args.bptt)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        if char_data is not None:
            char_data = char_data.as_in_context(ctx)
        output, _, _, _, masks = model(data, char_data, valid_length=valid_lengths, masks=masks)
        L = loss(output.reshape(-3, -1),
                 target.reshape(-1,))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
        i += args.bptt
    return total_L / ntotal

def evaluate_with_cache(cache_cell, data_source, char_data_source, batch_size, ctx=None):
    """Evaluate the model on the dataset with cache model.

    Parameters
    ----------
    data_source : NDArray
        The dataset is evaluated on.
    batch_size : int
        The size of the mini-batch.
    ctx : mx.cpu() or mx.gpu()
        The context of the computation.

    Returns
    -------
    loss: float
        The loss on the dataset
    """
    total_L = 0
    next_word_history = None
    cache_history = None
    i = 0
    while i < len(data_source) - 1 - 1:
        if i > 0 and i % (100 * args.cache_bptt) == 0:
            print('Batch %d/%d, ppl %f'%
                  (i, len(data_source), math.exp(total_L/i)))
        data, target, char_data, _ = get_batch(data_source, char_data_source, i, seq_len=args.cache_bptt)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        if args.use_cnn_embedding:
            char_data = char_data.as_in_context(ctx)
        L = 0
        outs, next_word_history, cache_history = \
            cache_cell(data, char_data, target, next_word_history, cache_history, valid_length=valid_lengths)
        for out in outs:
            L += (-mx.nd.log(out)).asscalar()
        total_L += L / data.shape[1]
        i += args.cache_bptt
    return total_L / len(data_source)

def warmup_lr(step_num):
    r"""Implement learning rate warm up.
         Parameters
        -----------
        step : int
            control the learning rate linear increase.
         Returns
        --------
        return : int
            the learning rate for next weight update.
        """
    return min(args.lr, args.lr * (math.log(step_num) / math.log(args.warmup_steps)))

def reset_embedding_grad(model):
    r"""
    Reset the grad about word embedding layer.
    """
    model.embedding.word_emb[0].weight.grad(ctx=context[0])[2:] = 0


def sync_multi_gpu_clip_global_norm(grads, context):
    for ctx in context:
        for _, (p, grad) in enumerate(zip(model.collect_params().values(), grads)):
            if p.grad_req != 'null':
                p.grad(ctx)[:] = grad.as_in_context(ctx)

def train():
    """

    """
    best_val = float('Inf')
    start_train_time = time.time()
    parameters = model.collect_params()
    step_num = 0
    number_params_flag = True

    ntasgd = False
    param_dict_avg = None
    t = 0
    avg_trigger = 0
    n = 5
    valid_losses = []
    masks = None
    for epoch in range(args.epochs):
        wc = 0
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        batch_i, i = 0, 0
        while i < len(train_data) - 1 - 1:
            step_num += 1
            if args.warmup_steps != 0:
                new_lr = warmup_lr(step_num)
                trainer.set_learning_rate(new_lr)
            seq_len = args.bptt
            data, target, char_data, char_target = get_batch(train_data, train_char_data, i, seq_len=seq_len)
            wc += data.shape[0]*data.shape[1]
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            if args.use_cnn_embedding:
                char_data_list = gluon.utils.split_and_load(char_data, context, batch_axis=1, even_split=True)
                char_target_list = gluon.utils.split_and_load(char_target, context, batch_axis=1, even_split=True)
            else:
                char_data_list = [None]*len(data_list)
                char_target_list = [None]*len(target_list)
            valid_length_list = gluon.utils.split_and_load(valid_lengths, context, batch_axis=0, even_split=True)
            Ls = []

            with autograd.record():
                for j, (X, y, CX, Cy, valid_len) in enumerate(zip(data_list, target_list, char_data_list, char_target_list, valid_length_list)):

                    output, _, encoder_hs, _, masks = model(X, CX, valid_length=valid_len, masks=masks)

                    if args.label_smoothing:
                        output = label_smoothing(output)
                    if number_params_flag:
                        total_params = sum(
                            v.data(context[0]).shape[0] if len(v.data(context[0]).shape) == 1 else v.data(context[0]).shape[1] if v.data(context[0]).shape[0] == 0 else v.data(context[0]).shape[0] if
                            v.data(context[0]).shape[1] == 0 else v.data(context[0]).shape[0] * v.data(context[0]).shape[1]
                            for k, v in model.collect_params().items())
                        emb_total_params = sum(
                            v.data(context[0]).shape[0] if len(v.data(context[0]).shape) == 1 else v.data(context[0]).shape[1] if v.data(context[0]).shape[0] == 0 else v.data(context[0]).shape[0] if
                            v.data(context[0]).shape[1] == 0 else v.data(context[0]).shape[0] * v.data(context[0]).shape[1]
                            for k, v in model.embedding.collect_params().items())
                        encoder_total_params = sum(
                            v.data(context[0]).shape[0] if len(v.data(context[0]).shape) == 1 else v.data(context[0]).shape[1] if v.data(context[0]).shape[0] == 0 else v.data(context[0]).shape[0] if
                            v.data(context[0]).shape[1] == 0 else v.data(context[0]).shape[0] * v.data(context[0]).shape[1]
                            for k, v in model.encoder.collect_params().items())
                        decoder_total_params = sum(
                            v.data(context[0]).shape[0] if len(v.data(context[0]).shape) == 1 else v.data(context[0]).shape[1] if v.data(context[0]).shape[0] == 0 else v.data(context[0]).shape[0] if
                            v.data(context[0]).shape[1] == 0 else v.data(context[0]).shape[0] * v.data(context[0]).shape[1]
                            for k, v in model.decoder.collect_params().items())
                        number_params_flag = False
                    l = loss(output, y)
                    Ls.append(l/X.size)
            for L in Ls:
                L.backward()

            if ema is not None and step_num == 1:
                for name, param in parameters.items():
                    ema.add(name, param.data(context[0]))

            trainer.allreduce_grads()

            if args.use_pretrained_embedding:
                if args.optimizer == 'adam':
                    reset_embedding_grad(model)
                    grads = []
                    for name, param in parameters.items():
                        if name == 'transformerlm0_transformerlm_enc_const':
                            grads.append(mx.nd.zeros((args.max_src_length, args.units)).as_in_context(context[0]))
                        else:
                            grad = param.grad(context[0])
                            if name == 'transformerlm0_hybridsequential0_embedding0_weight':
                                grad[0:2] += args.wd * \
                                             param.data(context[0])[0:2]
                            else:
                                grad += args.wd * param.data(context[0])
                            grads.append(grad)
                    gluon.utils.clip_global_norm(grads, args.clip)
                    reset_embedding_grad(model)
                elif args.optimizer == 'sgd':
                    reset_embedding_grad(model)
                    grads = []
                    for name, param in parameters.items():
                        if name == 'transformerlm0_transformerlm_enc_const':
                            grads.append(mx.nd.zeros((args.max_src_length, args.units)).as_in_context(context[0]))
                        else:
                            grads.append(param.grad(context[0]))
                    gluon.utils.clip_global_norm(grads, args.clip)
                    reset_embedding_grad(model)
                    for name, param in parameters.items():
                        if name != 'transformerlm0_transformerlm_enc_const':
                            if name == 'transformerlm0_hybridsequential0_embedding0_weight':
                                param.grad(context[0])[0:2] = trainer.learning_rate * \
                                                              (param.grad(context[0])[0:2] + args.wd * param.data(context[0])[0:2])
                            else:
                                param.grad(context[0])[:] = trainer.learning_rate * \
                                                            (param.grad(context[0])[:] + args.wd * param.data(context[0]))

                    #Add one more clipping trick
                    grads = []
                    for name, param in parameters.items():
                        if name == 'transformerlm0_transformerlm_enc_const':
                            grads.append(mx.nd.zeros((args.max_src_length, args.units)).as_in_context(context[0]))
                        else:
                            grads.append(param.grad(context[0]))
                    gluon.utils.clip_global_norm(grads, args.clip)

                    reset_embedding_grad(model)

            else:
                if len(context) == 1:
                    grads = [p.grad(d.context) if p.grad_req != 'null'
                             else mx.nd.zeros((args.max_src_length, args.units)).as_in_context(d.context)
                             for p in parameters.values()
                             for d in data_list]
                    gluon.utils.clip_global_norm(grads, args.clip)
                else:
                    #TODO: focus on clip_global_norm works for multiple gpu in not using pretrained embedding
                    grads = [(p.grad(context[0])/len(context)) if p.grad_req != 'null'
                             else mx.nd.zeros((args.max_src_length, args.units)).as_in_context(context[0])
                             for p in parameters.values()]
                    gluon.utils.clip_global_norm(grads, args.clip)
                    sync_multi_gpu_clip_global_norm(grads, context)
            if args.ntasgd and ntasgd:
                if param_dict_avg is None:
                    param_dict_avg = {k.split(model._prefix)[1]: v.data(context[0]).copy()
                                      for k, v in parameters.items()}

            trainer.update(1, ignore_stale_grad=True)

            if ema is not None:
                for name, param in parameters.items():
                    ema(name, param.data(context[0]))

            total_L += sum([mx.nd.sum(L).asscalar() for L in Ls]) / len(context)

            if batch_i % args.log_interval == 0 and batch_i > 0:
                cur_L = total_L / args.log_interval
                print('[Epoch %d Batch %d/%d] loss %.2f, '
                      'throughput %.2f samples/s, lr %f, wps=%.2f K, wc=%.2f K'
                      %(epoch, batch_i, len(train_data)//args.bptt, cur_L,
                        args.batch_size*args.log_interval/(time.time()-start_log_interval_time),
                        trainer.learning_rate,
                        wc / ((time.time()-start_log_interval_time)*1000), wc / 1000))
                total_L = 0.0
                start_log_interval_time = time.time()
                wc = 0
            i += seq_len
            batch_i += 1

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s'%(
            epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))
        if args.ntasgd and ntasgd:
            gamma = 1.0 / max(1, epoch - avg_trigger + 1)
            print('gamma: %f' % gamma)
            for name, param_avg in param_dict_avg.items():
                param_avg[:] += gamma * (parameters['{}{}'.format(model._prefix, name)]
                                         .data(context[0]) - param_avg)
            mx.nd.save('{}.val.params'.format(args.save), param_dict_avg)
        else:
            model.save_parameters('{}.val.params'.format(args.save))
        model.save_parameters('{}.{}'.format(args.save, 'tmp'))

        val_L = evaluate(val_data, val_char_data, val_batch_size, '{}.val.params'.format(args.save), context[0], ema)
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))

        model.load_parameters('{}.{}'.format(args.save, 'tmp'), context)

        if args.ntasgd and avg_trigger == 0:
            if t > n and val_L > min(valid_losses[-n:]):
                if param_dict_avg is None:
                    param_dict_avg = {k.split(model._prefix)[1]: v.data(context[0]).copy()
                                      for k, v in parameters.items()}
                else:
                    for k, v in parameters.items():
                        param_dict_avg[k.split(model._prefix)[1]] \
                            = v.data(context[0]).copy()
                avg_trigger = epoch
                print('Switching to NTASGD and avg_trigger is : %d' % avg_trigger)
                ntasgd = True
            valid_losses.append(val_L)
            t += 1

        if val_L < best_val:
            update_lr_epoch = 0
            best_val = val_L
            if args.ntasgd and ntasgd:
                mx.nd.save(args.save, param_dict_avg)
            else:
                model.save_parameters(args.save)
            test_L = evaluate(test_data, test_char_data, test_batch_size, args.save, context[0], ema)
            print('[Epoch %d] test loss %.2f, test ppl %.2f'
                  % (epoch, test_L, math.exp(test_L)))
        else:
            update_lr_epoch += 1
            if update_lr_epoch % args.lr_update_interval == 0 and update_lr_epoch != 0:
                lr_scale = trainer.learning_rate * args.lr_update_factor
                print('Learning rate after interval update %f'%(lr_scale))
                trainer.set_learning_rate(lr_scale)
                update_lr_epoch = 0

    print('Total training throughput %.2f samples/s'
          %((args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))


if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    if not args.use_cache:
        model.load_parameters(args.save, context)
        final_val_L = evaluate(val_data, val_char_data, val_batch_size, args.save, context[0], ema)
        final_test_L = evaluate(test_data, test_char_data, test_batch_size, args.save, context[0], ema)
        print('Best validation loss %.2f, val ppl %.2f'%(final_val_L, math.exp(final_val_L)))
        print('Best test loss %.2f, test ppl %.2f'%(final_test_L, math.exp(final_test_L)))
    else:
        cache_cell = CacheCell(model, ntokens, args.window, args.theta, args.lambdas, args.cache_output)
        cache_cell.load_parameters(args.path_to_params_file, ctx=context)
        final_val_L_cache = evaluate_with_cache(cache_cell, val_data, val_char_data, val_batch_size, context[0])
        final_test_L_cache = evaluate_with_cache(cache_cell, test_data, test_char_data, test_batch_size, context[0])
        print('Best cache validation loss %.2f, val ppl %.2f'%(final_val_L_cache, math.exp(final_val_L_cache)))
        print('Best cache test loss %.2f, test ppl %.2f'%(final_test_L_cache, math.exp(final_test_L_cache)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
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
"""."""
__all__ = ['']

import math
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet import init, nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.block import Block, HybridBlock
try:
    from encoder_decoder_language_model import Seq2SeqEncoder, _get_attention_cell
except ImportError:
    from .encoder_decoder_language_model import Seq2SeqEncoder, _get_attention_cell


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


def _windowed_mask(mask, window_size):
    if mask is None:
        return mask
    for i in range(mask.shape[0]):
        mask[i][max(0, i-(int(window_size)-1)):min(mask.shape[1], i+1)] = 1
    return mask


def _dilated_windowed_mask(mask, l, k=3, d_base=2):
    if mask is None:
        return mask
    d = math.pow(d_base, l)
    for i in range(mask.shape[0]-1, -1, -1):
        end_idx = int(i)
        mask[i][end_idx] = 1
        for j in range(k-1):
            end_idx = int(end_idx - d)
            if end_idx < 0:
                break
            mask[i][end_idx] = 1
    return mask

def _dilated_mem_windowed_mask(mask, l, k=3, d_base=2):
    if mask is None:
        return mask
    d = math.pow(d_base, l)
    for i in range(mask.shape[0]-1, -1, -1):
        end_idx = int(i)
        mask[i][end_idx] = 1
        for j in range(k-1):
            end_idx = int(end_idx - d)
            if end_idx < 0:
                break
            mask[i][end_idx] = 1
    return mask

class PositionwiseFFN(HybridBlock):
    """Structure of the Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 use_encoder_ffn_variational_dropout=False,
                 use_layer_norm=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        self._use_encoder_ffn_variational_dropout = use_encoder_ffn_variational_dropout
        self._use_layer_norm = use_layer_norm
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            if self._use_encoder_ffn_variational_dropout:
                self.dropout_layer = nn.Dropout(dropout, axes=(1,))
            else:
                self.dropout_layer = nn.Dropout(dropout)
            if self._use_layer_norm:
                self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Position-wise encoding of the inputs.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)

        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        if self._use_layer_norm:
            outputs = self.layer_norm(outputs)
        return outputs


class TransformerEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_drop=0.5,
                 use_weight_dropout=True,
                 use_encoder_cell_variational_dropout=False,
                 use_encoder_ffn_variational_dropout=False,
                 use_layer_norm=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._weight_drop = weight_drop
        self._use_weight_dropout = use_weight_dropout
        self._use_encoder_cell_variational_dropout = use_encoder_cell_variational_dropout
        self._use_encoder_ffn_variational_dropout = use_encoder_ffn_variational_dropout
        self._use_layer_norm = use_layer_norm

        with self.name_scope():
            if self._use_encoder_cell_variational_dropout:
                self.dropout_layer = nn.Dropout(dropout, axes=(1,))
            else:
                self.dropout_layer = nn.Dropout(dropout)
            if self._use_weight_dropout:
                self.attention_cell = _get_attention_cell(attention_cell,
                                                          units=units,
                                                          num_heads=num_heads,
                                                          scaled=scaled,
                                                          dropout=weight_drop,
                                                          use_weight_dropout=True)
            else:
                self.attention_cell = _get_attention_cell(attention_cell,
                                                          units=units,
                                                          num_heads=num_heads,
                                                          scaled=scaled,
                                                          dropout=dropout,
                                                          use_weight_dropout=False)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       use_encoder_ffn_variational_dropout=use_encoder_ffn_variational_dropout,
                                       use_layer_norm=use_layer_norm,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            if self._use_layer_norm:
                self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights =\
            self.attention_cell(inputs, inputs, inputs, mask)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        if self._use_layer_norm:
            outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs




class TransformerEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the Transformer Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, model='full', attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 window_size=2, window_size_multiplier=2,
                 weight_drop=0.5,
                 use_weight_dropout=True,
                 use_encoder_variational_dropout=False,
                 use_encoder_cell_variational_dropout=False,
                 use_encoder_ffn_variational_dropout=False,
                 use_layer_norm=True,
                 kernel_size=3,
                 d_base=2,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._model = model
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        self._window_size = window_size
        self._window_size_multiplier = window_size_multiplier
        self._weight_drop = weight_drop
        self._use_weight_dropout = use_weight_dropout
        self._use_encoder_variational_dropout = use_encoder_variational_dropout
        self._use_encoder_cell_variational_dropout = use_encoder_cell_variational_dropout
        self._use_encoder_ffn_variational_dropout = use_encoder_ffn_variational_dropout
        self._use_layer_norm = use_layer_norm
        self._kernel_size = kernel_size
        self._d_base = d_base

        with self.name_scope():
            if self._use_encoder_variational_dropout:
                self.dropout_layer = nn.Dropout(dropout, axes=(1,))
            else:
                self.dropout_layer = nn.Dropout(dropout)
            if self._use_layer_norm:
                self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout if i != num_layers - 1 else 0.0,
                        use_residual=use_residual,
                        scaled=scaled,
                        output_attention=output_attention,
                        weight_drop=self._weight_drop,
                        use_weight_dropout=self._use_weight_dropout,
                        use_encoder_cell_variational_dropout=self._use_encoder_cell_variational_dropout,
                        use_encoder_ffn_variational_dropout=self._use_encoder_ffn_variational_dropout,
                        use_layer_norm=self._use_layer_norm,
                        prefix='transformer%d_' % i))

    def __call__(self, inputs, states=None, valid_length=None, masks=None): #pylint: disable=arguments-differ
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states and masks
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super(TransformerEncoder, self).__call__(inputs, states, valid_length, masks)

    def forward(self, inputs, states=None, valid_length=None, masks=None, steps=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray, Shape(batch_size, length, C_in)
        states : list of NDArray
        valid_length : NDArray
        steps : NDArray
            Stores value [0, 1, ..., length].
            It is used for lookup in positional encoding matrix

        Returns
        -------
        outputs : NDArray
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        #TNC->NTC
        inputs = inputs.swapaxes(dim1=0, dim2=1)
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        length_array = mx.nd.arange(length, ctx=inputs.context)

        if masks is None or length != masks[0].shape[1]:
            masks = []
            for i in range(self._num_layers):
                if self._window_size == -1 or self._model == 'full':
                    mask = mx.nd.broadcast_lesser_equal(
                        length_array.reshape((1, -1)),
                        length_array.reshape((-1, 1)))
                elif self._model == 'cascade':
                    window_size_i = self._window_size * math.pow(self._window_size_multiplier, i)
                    mask = mx.nd.zeros((length, length), ctx=inputs.context)
                    mask = _windowed_mask(mask, window_size_i)
                elif self._model == 'dilated':
                    mask = mx.nd.zeros((length, length), ctx=inputs.context)
                    mask = _dilated_windowed_mask(mask, i, self._kernel_size, self._d_base)
                elif self._model == 'dilated_mem':
                    if i == 0:
                        mask_prev = mx.nd.zeros((length, length), ctx=inputs.context)
                    mask = _dilated_mem_windowed_mask(mask_prev, i, self._kernel_size, self._d_base)
                    mask_prev = mask.copy()
                else:
                    raise NotImplementedError
                if valid_length is not None:
                    batch_mask = mx.nd.broadcast_lesser(
                        mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                        valid_length.reshape((-1, 1)))
                    mask = mx.nd.broadcast_mul(mx.nd.expand_dims(batch_mask, -1),
                                               mx.nd.expand_dims(mask, 0))
                else:
                    mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
                masks.append(mask)

        inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        if valid_length is not None:
            step_output, additional_outputs, encoded_raw =\
                super(TransformerEncoder, self).forward(inputs, states, masks)
        else:
            step_output, additional_outputs, encoded_raw =\
                super(TransformerEncoder, self).forward(inputs, states)
        return step_output, additional_outputs, encoded_raw, masks

    def hybrid_forward(self, F, inputs, states=None, masks=None, position_weight=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        if self._use_layer_norm:
            inputs = self.layer_norm(inputs)
        outputs = inputs
        additional_outputs = []
        encoded_raw = []
        for i, cell in enumerate(self.transformer_cells):
            outputs, attention_weights = cell(inputs, masks[i])
            encoded_raw.append(outputs.transpose(axes=(1,0,2)))
            inputs = outputs
            if self._output_attention:
                additional_outputs.append(attention_weights)
        return outputs, additional_outputs, encoded_raw

def get_transformer_encoder(model='full', num_layers=2,
                            num_heads=8, scaled=True,
                            units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                            max_src_length=50,
                            window_size=2,
                            window_size_multiplier=2,
                            weight_drop=0.5,
                            use_weight_dropout=True,
                            use_encoder_variational_dropout=False,
                            use_encoder_cell_variational_dropout=False,
                            use_encoder_ffn_variational_dropout=False,
                            use_layer_norm=True,
                            kernel_size=3,
                            d_base=2,
                            weight_initializer=None, bias_initializer='zeros',
                            prefix='transformer_', params=None):
    """Build a pair of Parallel GNMT encoder/decoder

    Parameters
    ----------
    num_layers : int
    num_heads : int
    scaled : bool
    units : int
    hidden_size : int
    dropout : float
    use_residual : bool
    max_src_length : int
    max_tgt_length : int
    weight_initializer : mx.init.Initializer or None
    bias_initializer : mx.init.Initializer or None
    prefix : str, default 'transformer_'
        Prefix for name of `Block`s.
    params : Parameter or None
        Container for weight sharing between layers.
        Created if `None`.

    Returns
    -------
    encoder : TransformerEncoder
    decoder :TransformerDecoder
    """
    encoder = TransformerEncoder(model=model, num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_src_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 use_residual=use_residual,
                                 window_size=window_size,
                                 window_size_multiplier=window_size_multiplier,
                                 weight_drop=weight_drop,
                                 use_weight_dropout=use_weight_dropout,
                                 use_encoder_variational_dropout=use_encoder_variational_dropout,
                                 use_encoder_cell_variational_dropout=use_encoder_cell_variational_dropout,
                                 use_encoder_ffn_variational_dropout=use_encoder_ffn_variational_dropout,
                                 use_layer_norm=use_layer_norm,
                                 kernel_size=kernel_size,
                                 d_base=d_base,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix=prefix + 'enc_', params=params)

    return encoder


class EmbeddingBlock(HybridBlock):
    def __init__(self, CORPUS_WORDS, CORPUS_CHARACTERS, LAYERS_DROPOUT=0.1, EMB_ENCODER_CONV_CHANNELS=128, NUM_HIGHWAY_LAYERS=2,
                  DIM_WORD_EMBED=300, WORD_EMBEDDING_DROPOUT=0.1, DIM_CHAR_EMBED=200, CHAR_EMBEDDING_DROPOUT=0.05,
                  EMB_ENCODER_CONV_KERNEL_SIZE=7, EMB_ENCODER_NUM_CONV_LAYERS=4, MAX_CHARACTER_PER_WORD=16, use_highway=True, use_cnn_embedding=True, **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self._LAYERS_DROPOUT = LAYERS_DROPOUT
        self._EMB_ENCODER_CONV_CHANNELS = EMB_ENCODER_CONV_CHANNELS
        self._NUM_HIGHWAY_LAYERS = NUM_HIGHWAY_LAYERS
        self._CORPUS_WORDS = CORPUS_WORDS
        self._DIM_WORD_EMBED = DIM_WORD_EMBED
        self._WORD_EMBEDDING_DROPOUT = WORD_EMBEDDING_DROPOUT
        self._CORPUS_CHARACTERS = CORPUS_CHARACTERS
        self._DIM_CHAR_EMBED = DIM_CHAR_EMBED
        self._CHAR_EMBEDDING_DROPOUT = CHAR_EMBEDDING_DROPOUT
        self._EMB_ENCODER_CONV_KERNEL_SIZE = EMB_ENCODER_CONV_KERNEL_SIZE
        self._EMB_ENCODER_NUM_CONV_LAYERS = EMB_ENCODER_NUM_CONV_LAYERS
        self._MAX_CHARACTER_PER_WORD = MAX_CHARACTER_PER_WORD
        self._use_highway = use_highway
        self._use_cnn_embedding = use_cnn_embedding
        if use_cnn_embedding:
            self._highway_dimension = DIM_CHAR_EMBED + DIM_WORD_EMBED
        else:
            self._highway_dimension = DIM_WORD_EMBED

        self.word_emb = gluon.nn.HybridSequential()
        with self.word_emb.name_scope():
            self.word_emb.add(
                gluon.nn.Embedding(
                    input_dim=CORPUS_WORDS,
                    output_dim=DIM_WORD_EMBED
                )
            )
            if self._WORD_EMBEDDING_DROPOUT != 0:
                self.word_emb.add(
                    gluon.nn.Dropout(rate=WORD_EMBEDDING_DROPOUT)
                )

        if self._use_cnn_embedding:
            self.char_emb = gluon.nn.HybridSequential()
            with self.char_emb.name_scope():
                # self.char_emb.add(
                #     gluon.nn.Embedding(
                #         input_dim=CORPUS_CHARACTERS,
                #         output_dim=DIM_CHAR_EMBED,
                #         weight_initializer=init.Normal(sigma=0.1)
                #     )
                # )

                self.char_emb.add(
                    gluon.nn.Embedding(
                        input_dim=CORPUS_CHARACTERS,
                        output_dim=8,
                        weight_initializer=init.Normal(sigma=0.1)
                    )
                )
                self.char_emb.add(
                    nlp.model.ConvolutionalEncoder(
                        embed_size=8,
                        num_filters=(100,),
                        ngram_filter_sizes=(5,),
                        num_highway=None,
                        output_size=self._DIM_CHAR_EMBED
                    ))
                if self._CHAR_EMBEDDING_DROPOUT != 0:
                    self.char_emb.add(
                        gluon.nn.Dropout(rate=CHAR_EMBEDDING_DROPOUT)
                    )

        if self._use_highway:
            self.highway = gluon.nn.HybridSequential()
            with self.highway.name_scope():
                self.highway.add(
                    nlp.model.Highway(
                        input_size=self._highway_dimension,
                        num_layers=NUM_HIGHWAY_LAYERS
                    )
                )

    def hybrid_forward(self, F, context, context_char=None):
        if self._use_cnn_embedding:
            context_word_emb = self.word_emb(context)
            context_char_emb = self.char_emb[0](context_char)
            context_char_emb = F.transpose(context_char_emb, axes=(0,2,1,3))
            char_emb_list = []
            for char_emb in context_char_emb:
                char_emb_list.append(self.char_emb[1](char_emb))
            context_char_emb = F.concat(*char_emb_list, dim=0)
            context_char_emb = F.reshape(context_char_emb, shape=(len(char_emb_list), -1, self._DIM_CHAR_EMBED))
            context_concat = F.concat(context_word_emb, context_char_emb, dim=-1)
            if self._use_highway:
                context_final_emb = self.highway(context_concat)
            else:
                context_final_emb = context_concat
        else:
            context_word_emb = self.word_emb(context)
            if self._use_highway:
                context_final_emb = self.highway(context_word_emb)
            else:
                context_final_emb = context_word_emb
        return context_final_emb


class TransformerLM(Block):
    """
    """
    def __init__(self, model, vocab_size, char_vocab_size, embed_size=512, char_embed_size=200, hidden_size=2048, num_layers=6,
                 tie_weights=True, dropout=0.4, weight_drop=0.5, drop_h=0.2,
                 drop_i=0.65, drop_e=0.1, num_heads=8, scaled=True, units=512, use_residual=True,
                 max_src_length=50, window_size=2, window_size_multiplier=2,
                 use_weight_dropout=True,
                 use_encoder_last_variational_dropout=True,
                 use_encoder_variational_dropout=False,
                 use_encoder_cell_variational_dropout=False,
                 use_encoder_ffn_variational_dropout=False,
                 use_layer_norm=True,
                 use_pretrained_embedding=True,
                 use_highway=True,
                 use_cnn_embedding=True,
                 max_word_length=16,
                 kernel_size=3,
                 d_base=2,
                 weight_initializer=None, bias_initializer='zeros', **kwargs):
        super(TransformerLM, self).__init__(**kwargs)
        self._model = model
        self._vocab_size = vocab_size
        self._char_vocab_size = char_vocab_size
        self._embed_size = embed_size
        self._char_embed_size = char_embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights
        self._num_heads = num_heads
        self._scaled = scaled
        self._units = units #TODO: equal with embed_Size?
        self._max_src_length = max_src_length ##TODO: = data.size(0) double check
        self._use_residual = use_residual
        self._window_size = window_size
        self._window_size_multiplier = window_size_multiplier
        self._use_weight_dropout = use_weight_dropout
        self._use_encoder_last_variational_dropout = use_encoder_last_variational_dropout
        self._use_encoder_variational_dropout = use_encoder_variational_dropout
        self._use_encoder_cell_variational_dropout = use_encoder_cell_variational_dropout
        self._use_encoder_ffn_variational_dropout = use_encoder_ffn_variational_dropout
        self._use_layer_norm = use_layer_norm
        self._use_pretrained_embedding = use_pretrained_embedding
        self._use_highway = use_highway
        self._use_cnn_embedding = use_cnn_embedding
        self._max_word_length = max_word_length
        self._kernel_size = kernel_size
        self._d_base = d_base
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        if self._use_pretrained_embedding:
            embedding = EmbeddingBlock(CORPUS_WORDS=self._vocab_size, CORPUS_CHARACTERS=self._char_vocab_size,
                                       DIM_WORD_EMBED=self._embed_size,
                                       use_highway=self._use_highway, WORD_EMBEDDING_DROPOUT=self._drop_i, DIM_CHAR_EMBED=self._char_embed_size,
                                       MAX_CHARACTER_PER_WORD=self._max_word_length)
        else:
            embedding = nn.HybridSequential()
            with embedding.name_scope():
                embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                               weight_initializer=init.Uniform(0.1))
                if self._drop_e:
                    nlp.model.utils.apply_weight_drop(embedding_block, 'weight', self._drop_e,
                                                      axes=(1,))
                embedding.add(embedding_block)
                if self._drop_i:
                    embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = get_transformer_encoder(model=self._model, num_layers=self._num_layers, num_heads=self._num_heads, scaled=self._scaled,
                                          units=self._units, hidden_size=self._hidden_size, dropout=self._drop_h,
                                          use_residual=self._use_residual, max_src_length=self._max_src_length,
                                          window_size=self._window_size, window_size_multiplier=self._window_size_multiplier,
                                          weight_drop=self._weight_drop,
                                          use_weight_dropout=self._use_weight_dropout,
                                          use_encoder_variational_dropout=self._use_encoder_variational_dropout,
                                          use_encoder_cell_variational_dropout=self._use_encoder_cell_variational_dropout,
                                          use_encoder_ffn_variational_dropout=self._use_encoder_ffn_variational_dropout,
                                          use_layer_norm=self._use_layer_norm,
                                          kernel_size=self._kernel_size,
                                          d_base=self._d_base,
                                          weight_initializer=self._weight_initializer,
                                          bias_initializer=self._bias_initializer,
                                          prefix='transformerlm_', params=None)
        return encoder

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def __call__(self, inputs, char_inputs, begin_state=None, valid_length=None, masks=None):  #pylint: disable=arguments-differ
        """Generate the prediction given the src_seq and tgt_seq.

        This is used in training an NMT model.

        Parameters
        ----------
        src_seq : NDArray
        tgt_seq : NDArray
        src_valid_length : NDArray or None
        tgt_valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, tgt_length, tgt_word_num)
        additional_outputs : list of list
            Additional outputs of encoder and decoder, e.g, the attention weights
        """
        return super(TransformerLM, self).__call__(inputs, char_inputs, begin_state, valid_length, masks)

    def forward(self, inputs, char_inputs, begin_state=None, valid_length=None, masks=None): # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        encoded_raw: list
            The list of outputs of the model's encoder with length equals to num_layers.
            the shape of every encoder's output `(sequence_length, batch_size, num_hidden)`
        encoded_dropped: list
            The list of outputs with dropout of the model's encoder with length equals
            to num_layers. The shape of every encoder's dropped output
            `(sequence_length, batch_size, num_hidden)`
        """
        if self._use_pretrained_embedding:
            encoded = self.embedding(inputs, char_inputs)
        else:
            encoded = self.embedding(inputs)
        encoded, _, encoded_raw, masks = self.encoder(encoded, valid_length=valid_length, masks=masks)
        if self._use_encoder_last_variational_dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(1,))
        else:
            encoded = nd.Dropout(encoded, p=self._dropout)
        encoded = encoded.swapaxes(dim1=0, dim2=1)
        out = self.decoder(encoded)
        return out, _, encoded_raw, _, masks

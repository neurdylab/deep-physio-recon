import torch
torch.manual_seed(11)
import torch.nn as nn
from typing import Optional, Union
import torch.nn.functional as F
import numpy as np
np.random.seed(11)

class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = None,
                 pe: str = None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding1 = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear1 = nn.Linear(d_model, d_output)

        self.layers_decoding2 = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._linear2 = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding1:
            decoding1 = layer(decoding, encoding)

        for layer in self.layers_decoding2:
            decoding2 = layer(decoding, encoding)

        # Output module
        output1 = self._linear1(decoding1)
        output2 = self._linear2(decoding2)
        return output1, output2

class Decoder(nn.Module):
    """Decoder block from Attention is All You Need.

    Apply two Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        """Initialize the Decoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._encoderDecoderAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Decoder block.

        Apply the self attention block, add residual and normalize.
        Apply the encoder-decoder attention block, add residual and normalize.
        Apply the feed forward network, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        memory:
            Memory tensor with shape (batch_size, K, d_model)
            from encoder output.

        Returns
        -------
        x:
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x, mask="subsequent")
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Encoder-decoder attention
        residual = x
        x = self._encoderDecoderAttention(query=x, key=memory, value=memory)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm3(x + residual)

        return x

class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        """Initialize the Encoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but identically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu activation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(self._linear1(x))

def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 64) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.300
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)

class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)
        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)

        # Compute local map mask
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # !!!!!!!!!!!!!!!!!!!!!!! WHAT TO DO WITH THIS !!!!!!!!!!!!!!!!!!!!!!!!
        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    chunk_size:
        Size of chunks to apply attention on. Last one may be smaller (see :class:`torch.Tensor.chunk`).
        Default is 168.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 chunk_size: Optional[int] = 168,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._chunk_size = chunk_size

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]

        n_chunk = K // self._chunk_size

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._chunk_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(
            n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention


class MultiHeadAttentionWindow(MultiHeadAttention):
    """Multi Head Attention block with moving window.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks using a moving window.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    window_size:
        Size of the window used to extract chunks.
        Default is 168
    padding:
        Padding around each window. Padding will be applied to input sequence.
        Default is 168 // 4 = 42.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 window_size: Optional[int] = 168,
                 padding: Optional[int] = 168 // 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._window_size = window_size
        self._padding = padding
        self._q = q
        self._v = v

        # Step size for the moving window
        self._step = self._window_size - 2 * self._padding

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, self._window_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._window_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        batch_size = query.shape[0]

        # Apply padding to input sequence
        query = F.pad(query.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        key = F.pad(key.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        value = F.pad(value.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Divide Q, K and V using a moving window
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._v, self._window_size)).transpose(1, 2)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._window_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Fold chunks back
        attention = attention.reshape((batch_size*self._h, -1, self._window_size, self._v))
        attention = attention[:, :, self._padding:-self._padding, :]
        attention = attention.reshape((batch_size*self._h, -1, self._v))

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention
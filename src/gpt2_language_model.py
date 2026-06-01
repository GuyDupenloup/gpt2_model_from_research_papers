# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention mechanism.
    Implements causal (autoregressive) attention with configurable number of heads.
    """
    def __init__(self, seq_len, d_model, n_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.seq_len = seq_len
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        # -inf constant giving 0.0 after softmax
        self.epsilon = tf.constant(-1e9, dtype=tf.float32)

        # Concatenated Wq, Wk and Wv matrices
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name='W_qkv')

        # Output projection matrix
        self.output_proj = tf.keras.layers.Dense(d_model, name='out_proj')

        self.causal_mask = tf.linalg.band_part(
            tf.ones((self.seq_len, self.seq_len), dtype=tf.bool), -1, 0
        )
        
        
    def call(self, inputs, attention_mask):

        # Get the batch size
        batch = tf.shape(inputs)[0]

        # Multiply inputs by query/key/value weight matrices
        QKV = self.W_qkv(inputs)

        # Separate Q/K/V
        # Shape: (batch, d_model)
        Q, K, V = tf.split(QKV, num_or_size_splits=3, axis=-1)

        # d_model = d_head * n_heads
        Q = tf.reshape(Q, (batch, self.seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, self.seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, self.seq_len, self.n_heads, self.d_head))

        # (batch, seq_len, n_heads, d_head) -> (batch, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Calculate dot products between queries and keys
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))

        # Scale the scores, apply the attention and causal mask,
        # and apply softmax to get the attention weights
        scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        attn_mask = tf.cast(attention_mask, tf.bool)
        scores = tf.where(attn_mask[:, None, None, :], scores, self.epsilon)

        scores = tf.where(self.causal_mask[None, None, :, :], scores, self.epsilon)

        attn_weights = tf.nn.softmax(scores, axis=-1)

        # Multiply the scaled attention scores by value weight matrix
        context = tf.matmul(attn_weights, V)

        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch, self.seq_len, self.d_model))

        # Output projection
        d_out = self.output_proj(context)

        return d_out


class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network.
    Applies two linear transformations with GELU activation.
    """
    def __init__(self, d_model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model

        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=tf.keras.activations.gelu,
            name='ffn_inner',
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

    def call(self, inputs, training=None):
        x = self.ff_inner(inputs)
        x = self.ff_out(x)
        return x


class GPT2Transformer(tf.keras.layers.Layer):
    """
    GPT2 transformer block.
    Consists of multi-head attention and feed-forward network,
    each with layer normalization and residual connections.
    """
    def __init__(
            self, seq_len, d_model, n_heads, dropout_rate=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_1')
        self.attn_heads = MultiHeadAttention(seq_len, d_model, n_heads, name='attn_heads')
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate, name='drop_1')
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_2')
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate, name='drop_2')


    def call(self, inputs, attention_mask, training=None):

        # First sub-layer
        x1 = self.layer_norm_1(inputs, training=training)
        x1 = self.attn_heads(x1, attention_mask=attention_mask)
        x1 = self.dropout_1(x1, training=training)

        # First residual connection
        x2 = x1 + inputs

        # Second sub-layer
        x3 = self.layer_norm_2(x2, training=training)
        x3 = self.ffn(x3, training=training)
        x3 = self.dropout_2(x3, training=training)

        # Second residual connection
        output = x2 + x3

        return output


class GPT2Model(tf.keras.models.Model):
    """
    Original OpenAI GPT-2 model.

    Arguments:
        model_config:
            The model configuration parameters, a dictionary 
            with the following items:
                "vocab_size": vocabulary size.
                "max_seq_len": input sequence maximum length (context size).
                "d_model": hidden state size (embeddings size).
                "n_layers": number of transformer blocks.
                "n_heads": number of attention heads.
            These parameters for a given model size can be obtained using
            the get_gpt2_model_config() function in model_utils.py.

        dropout_rate:
            The dropout rate for all the dropout layers of the model, a float >= 0.
    """


    def __init__(self, model_config, dropout_rate=0., name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.config = model_config
        self.dropout_rate = dropout_rate

        # Get model config parameters
        vocab_size, seq_len, d_model, n_layers, n_heads = (
            model_config[k] for k in ('vocab_size', 'seq_len', 'd_model', 'n_layers', 'n_heads')
        )
        self.seq_len = seq_len

        # Token and position embedding layers
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='tkn_emb')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='pos_emb')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Transformer layers
        self.transformer_layers = [
            GPT2Transformer(
                seq_len,
                d_model,
                n_heads,
                dropout_rate=dropout_rate,
                name=f'transformer_{i}'
            ) 
            for i in range(n_layers)
        ]

        self.layer_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')


    def call(self, inputs, attention_mask, training=None):
        """
        Forward pass through the GPT-2 model.

        Arguments:
            inputs:
                Input token sequences.
                A tensor with shape (batch, seq_len).

            attention_mask:
                Mask specifying which token positions to attend to.
                A tensor with shape (batch, seq_len).

            training:
                Training or evaluation mode.

        Returns:
            The hidden state output, a tensor with shape (batch, seq_len, d_model).
        """

        # Token embeddings
        token_embed = self.token_embed_layer(inputs)

        # Position embeddings
        position_ids = tf.range(self.seq_len)
        position_embed = self.position_embed_layer(position_ids)

        # Embeddings
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for transformer in self.transformer_layers:
            x = transformer(x, attention_mask, training=training)

        output = self.layer_norm_final(x, training=training)

        return output


class GPT2LanguageModel(tf.keras.models.Model):
    """
    GPT-2 language modelling head.

    See docstring of GPT2Model.
    """

    def __init__(self, model_config, dropout_rate=0., name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = model_config
        self.gpt2_model = GPT2Model(model_config, dropout_rate=dropout_rate, name='gpt2_model')


    def call(self, inputs, training=None):
        """
        Forward pass through the GPT-2 language model.

        See docstring of GPT2Model.call().
        """
        
        input_ids = inputs['input_ids']       # Input sequence token IDs
        attention_mask = inputs['attention_mask']

        gpt2_output = self.gpt2_model(input_ids, attention_mask, training=training)

        # Output linear layer that projects hidden state representations to vocabulary.
        # Weights of the projection matrix are shared with the token embedding matrix.
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits

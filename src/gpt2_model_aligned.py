# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

MODEL_CONFIGS = {
    'gpt2':        {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
    'gpt2-medium': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
    'gpt2-large':  {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
    'gpt2-xl':     {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
}


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        # Concatenated Wq, Wk and Wv matrices
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name='W_qkv')

        # Output projection matrix
        self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')


    def call(self, input, attention_mask=None, training=False):

        batch, seq_len, _ = tf.unstack(tf.shape(input))

        # Get queries, keys and values
        QKV = self.W_qkv(input)
        Q, K, V = tf.split(QKV, num_or_size_splits=3, axis=-1)

        # d_model = d_head * n_heads
        Q = tf.reshape(Q, (batch, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, seq_len, self.n_heads, self.d_head))

        # Transpose from (batch, seq_len, n_heads, d_head)
        # to (batch, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Calculate the attention scores (dot products between queries and keys)
        # Shape: (batch, seq_len, d_model)
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))

        epsilon = tf.constant(-1e9, dtype=tf.float32)
        if attention_mask is not None:
            # Apply the attention mask (mask out padding tokens in keys)
            attn_mask = attention_mask[:, None, None, :]  # Broadcast to (batch, 1, 1, seq_len) to mask keys
            scores = tf.where(attn_mask == 0, epsilon, scores)

        # Apply causal attention using a triangular matrix
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        scores = tf.where(causal_mask[None, None, :, :], scores, epsilon)

        # Scale the scores and apply softmax to get the attention weights
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        attn_weights = tf.nn.softmax(scaled_scores, axis=-1)

        # Calculate the context vectors
        # shape: (batch, n_heads, seq_len, d_head)
        context = tf.matmul(attn_weights, V)

        # Transpose to have (batch, seq_len, n_heads, d_head)
        context = tf.transpose(context, perm=[0, 2, 1, 3])

        # Reshape to output size
        context = tf.reshape(context, (batch, seq_len, self.d_model))

        # Output projection layer
        d_out = self.c_proj(context)

        return d_out


class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.ff_inner = tf.keras.layers.Dense(4 * d_model, activation=tf.keras.activations.gelu, name='ffn_inner')
        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

    def call(self, input, training=False):
        x = self.ff_inner(input)
        x = self.ff_out(x)
        return x


class GPT2Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # 1st LayerNorm layer
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_1')
        
        # Multi-head attention block
        self.attention = MultiHeadAttention(d_model, n_heads, name='attention')
        
        # 2nd LayerNorm layer
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_2')

        # Feedforward network
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')

        # Dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)


    def call(self, input, attention_mask=None, training=False):

        input_norm = self.norm_1(input)
        attn_out = self.attention(input_norm, attention_mask=attention_mask, training=training)
        attn_out = self.dropout_1(attn_out, training=training)

        # Residual connection
        x = attn_out + input

        x_norm = self.norm_2(x)
        ff_out = self.ffn(x_norm, training=training)
        ff_out = self.dropout_2(ff_out, training=training)

        # Residual connection
        output = x + ff_out

        return output


class GPT2Model(tf.keras.models.Model):
    """
        Arguments:
            model_size:
                Size of the model, one of ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl').
            dropout_rate:
                Dropout rate for dropout layers (defaults to 0.1)

        Returns:
            Logits over vocabulary
            A tensor of floats with shape (batch_size, seq_len, vocabulary)
    """

    def __init__(self, model_size, dropout_rate=0.1, **kwargs):
        super().__init__(name=model_size, **kwargs)
        
        # Get model config parameters
        config = MODEL_CONFIGS[model_size]
        vocab_size, seq_len, d_model, n_layers, n_heads = (
            config[k] for k in ('vocab_size', 'seq_len', 'd_model', 'n_layers', 'n_heads')
        )

        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embd')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='position_embd')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
        self.transformer_blocks = [
            GPT2Transformer(d_model, n_heads, dropout_rate, name=f'transformer_{i}') 
            for i in range(n_layers)
        ]

        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')
    

    def call(self, input, attention_mask=None, training=False):
        seq_len = tf.shape(input)[1]
        
        token_embed = self.token_embed_layer(input)

        # Add position embeddings
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embed = self.position_embed_layer(positions)  # Shape: (seq_len, d_model)
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask, training=training)

        x = self.norm_f(x)

        # Get the embedding matrix We. Shape: (vocab_size, d_model)
        We = self.token_embed_layer.embeddings

        # Output layer: reuse token embedding weights for projection matrix W_o
        logits = tf.matmul(x, We, transpose_b=True)

        return logits

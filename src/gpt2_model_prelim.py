
import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        # Matrices for all heads
        self.W_q = tf.keras.layers.Dense(d_model, name='W_q')
        self.W_k = tf.keras.layers.Dense(d_model, name='W_k')
        self.W_v = tf.keras.layers.Dense(d_model, name='W_v')

        # Output projection
        self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')


    def call(self, input, training=False):

        # Get the batch size
        batch, seq_len, _ = tf.unstack(tf.shape(input))

        Q = self.W_q(input)
        K = self.W_k(input)
        V = self.W_v(input)

        # d_model = d_head * n_heads
        Q = tf.reshape(Q, (batch, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, seq_len, self.n_heads, self.d_head))

        # Transpose from (batch, seq_len, n_heads, d_head) to (batch, n_heads, seq_len, d_head)
        # to be able to access heads
        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Calculate the attention scores (dot products between queries and keys)
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))

        # Create a triangular mask for causal attention
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        epsilon = tf.constant(-1e9, dtype=scores.dtype)
        scores = tf.where(mask[None, None, :, :], scores, epsilon)

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
    def __init__(self, d_model, n_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # LayerNorm layers
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_1')
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_2')
        
        # Multi-head attention block
        self.attention = MultiHeadAttention(d_model, n_heads, name='attention')
        
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')

        # Dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.1)


    def call(self, input, training=False):

        input_norm = self.norm_1(input)

        attn_out = self.attention(input_norm, training=training)

        attn_out = self.dropout_1(attn_out, training=training)

        # Residual connection
        x = attn_out + input

        x_norm = self.norm_2(x)

        ff_out = self.ffn(x_norm)

        ff_out = self.dropout_2(ff_out, training=training)

        # Residual connection
        output = x + ff_out

        return output


class GPT2Model(tf.keras.layers.Layer):

    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embed')
        self.position_embed_layer = tf.keras.layers.Embedding(max_seq_len, d_model, name='position_embed')

        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        
        self.transformer_blocks = [
            GPT2Transformer(d_model, n_heads, name=f'transformer_{i}') 
            for i in range(n_layers)
        ]

        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')
    

    def call(self, input, training=False):
        seq_len = tf.shape(input)[1]
        
        token_embed = self.token_embed_layer(input)
        
        # Add position embeddings
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = token_embed + self.position_embed_layer(positions)

        x = self.dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm_f(x)

        # Output layer: reuse token embedding weights for projection matrix W_o
        logits = tf.matmul(x, token_embed, transpose_b=True)
        
        output = tf.nn.softmax(logits, axis=-1)

        return output

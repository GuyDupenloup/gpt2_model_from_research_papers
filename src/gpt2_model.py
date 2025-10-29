
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads


        # Output projection matrix
        self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')

        # Concatenated Wq, Wk and Wv matrices
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name='W_qkv')

        # Output projection matrix
        # self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')


    def call(self, input, training=False):

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

        # Apply causal attention using a triangular matrix
        epsilon = tf.constant(-1e9, dtype=tf.float32)
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

        # self.ff_inner = tf.keras.layers.Dense(4 * d_model, activation=tf.keras.activations.gelu, name='ffn_inner')
        # self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')
        self.ff_inner = tf.keras.layers.Dense(4 * d_model, activation=tf.keras.activations.gelu, name='ffn_inner')

    def call(self, input, training=False):
        x = self.ff_inner(input)
        x = self.ff_out(x)
        return x


class GPT2Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        '''
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
        '''
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_2')
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_1')
        self.attention = MultiHeadAttention(d_model, n_heads, name='attention')


    def call(self, input, training=False):

        input_norm = self.norm_1(input)
        attn_out = self.attention(input_norm, training=training)
        attn_out = self.dropout_1(attn_out, training=training)

        # Residual connection
        x = attn_out + input

        x_norm = self.norm_2(x)
        ff_out = self.ffn(x_norm, training=training)
        ff_out = self.dropout_2(ff_out, training=training)

        # Residual connection
        output = x + ff_out

        return output


class GPT2Model(tf.keras.layers.Layer):

    def __init__(self, vocab_size, seq_len, d_model, n_heads, n_layers, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        '''
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embd')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='position_embd')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
        self.transformer_blocks = [
            GPT2Transformer(d_model, n_heads, dropout_rate, name=f'transformer_{i}') 
            for i in range(n_layers)
        ]

        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')
        '''

        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')

        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embd')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.transformer_blocks = [
            GPT2Transformer(d_model, n_heads, dropout_rate, name=f'transformer_{i}') 
            for i in range(n_layers)
        ]

        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='position_embd')

    def call(self, input, training=False):
        seq_len = tf.shape(input)[1]
        
        token_embed = self.token_embed_layer(input)

        # Add position embeddings
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embed = self.position_embed_layer(positions)  # Shape: (seq_len, d_model)
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm_f(x)

        # Get the embedding matrix We. Shape: (vocab_size, d_model)
        We = self.token_embed_layer.embeddings

        # Output layer: reuse token embedding weights for projection matrix W_o
        logits = tf.matmul(x, We, transpose_b=True)

        return logits

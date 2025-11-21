import numpy as np
import tensorflow as tf
import tiktoken
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

GPT2_MODEL_CONFIGS = {
    'gpt2':        {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
    'gpt2-medium': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
    'gpt2-large':  {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
    'gpt2-xl':     {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
}


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, seq_len, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        # Concatenated Wq, Wk and Wv matrices
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name='W_qkv')

        # Output projection matrix
        self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')

        self.epsilon = 1e-9

        causal_mask = 1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        self.causal_mask = tf.cast(causal_mask[tf.newaxis, tf.newaxis, :, :], tf.float32)


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
        scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        if attention_mask is not None:
            # HF GPT-2 adds mask as large negative bias
            attn_mask = tf.cast((1 - attention_mask), tf.float32) * self.epsilon
            scores += attn_mask[:, None, None, :]

        # Apply causal attention using a triangular matrix (negative bias)
        causal_mask = 1.0 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        scores += self.causal_mask * self.epsilon
        attn_weights = tf.nn.softmax(scores, axis=-1)

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

        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=lambda x: tf.nn.gelu(x, approximate=True),
            name='ffn_inner'
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

    def call(self, input, training=False):
        x = self.ff_inner(input)
        x = self.ff_out(x)
        return x


class GPT2Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, seq_len, dropout_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # 1st LayerNorm layer
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_1')

        # Multi-head attention block
        self.attention = MultiHeadAttention(d_model, n_heads, seq_len, name='attention')

        # 2nd LayerNorm layer
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_2')

        # Feedforward network
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')

        # Dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)


    def call(self, input, attention_mask, training=False):

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
            dropout_rate:
                Dropout rate for dropout layers (defaults to 0.1)

        Returns:
            Logits over vocabulary
            A tensor of floats with shape (batch_size, seq_len, vocabulary)
    """

    def __init__(self, model_config, dropout_rate=0.1, name=None, **kwargs):
        if name is not None:
            super().__init__(name=name, **kwargs)
        else:
            super().__init__(**kwargs)

        # Get model config parameters
        self.config = model_config
        vocab_size, seq_len, d_model, n_layers, n_heads = (
            model_config[k] for k in ('vocab_size', 'seq_len', 'd_model', 'n_layers', 'n_heads')
        )
        self.seq_len = seq_len

        # Token and position embeddings
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embd')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='position_embd')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Transformer stack
        self.transformer_blocks = [
            GPT2Transformer(d_model, n_heads, seq_len, dropout_rate, name=f'transformer_{i}')
            for i in range(n_layers)
        ]

        # Final LayerNorm
        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')


    def call(self, data, training=False):

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        # Token embeddings
        token_embed = self.token_embed_layer(input_ids)
        self.embedding_weights = token_embed

        # Position embeddings
        pos_ids = tf.expand_dims(tf.range(self.seq_len), axis=0)
        position_embed = self.position_embed_layer(pos_ids)

        # Embeddings
        x = token_embed + position_embed

        x = self.dropout(x, training=training)

        # Transformer stack
        for block in self.transformer_blocks:
            x = block(x, attention_mask, training=training)

        output = self.norm_f(x)

        return output


class GPT2TextGenModel(tf.keras.models.Model):

    def __init__(self, model_config, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = model_config
        self.gpt2_model = GPT2Model(model_config, dropout_rate=dropout_rate, name=name)

        # Loss and metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy_tracker = tf.keras.metrics.Mean(name='accuracy')
        self.perplexity_tracker =  tf.keras.metrics.Mean(name='perplexity')


    def call(self, data):

        # data = {'input_ids': input_ids, 'attention_mask': attention_mask}
        gpt2_output = self.gpt2_model(data)

        # Text prediction linear layer (no trainable weights)
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits


    def compute_loss(self, input_ids, y_pred, mask):

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate predictions and shift mask to align with labels
        y_pred = y_pred[:, :-1, :]
        mask = mask[:, 1:]

        # Calculate cross-entropy loss per token (element wise)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        # Apply mask
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask

        # Return mean loss over non-masked tokens
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


    def compute_accuracy(self, input_ids, y_pred, mask):

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate predictions and shift mask to align with labels
        y_pred = y_pred[:, :-1, :]
        mask = mask[:, 1:]

        # Get predicted token IDs (argmax over vocabulary dimension)
        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # Shape: (batch_size, seq_len - 1)

        # Compare predictions with true labels (1.0 if correct, 0.0 if wrong)
        correct = tf.cast(tf.equal(predictions, y_true), dtype=tf.float32)

        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        correct_masked = correct * mask

        # Calculate accuracy: correct predictions / total non-masked tokens
        accuracy = tf.reduce_sum(correct_masked) / tf.maximum(tf.reduce_sum(mask), 1.0)

        return accuracy


    def train_step(self, data):
        # There are no labels as they are obtained by shifting the inputs.
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)  # Pass full x dict to call()
            loss = self.compute_loss(input_ids, y_pred, loss_mask)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.math.exp(loss)

        # Update loss and metrics trackers
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.perplexity_tracker.update_state(perplexity)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']

        y_pred = self(data, training=False)
        loss = self.compute_loss(input_ids, y_pred, loss_mask)

        # Compute metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.math.exp(loss)

        # Update loss and metrics trackers
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.perplexity_tracker.update_state(perplexity)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}


    # Register trackers
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker, self.perplexity_tracker]


def get_gpt2_model(model_name, pretrained=True):

    print(f'>> Creating `GPT2TextGenModel` model')

    model_config = GPT2_MODEL_CONFIGS[model_name]

    model = GPT2TextGenModel(model_config, name='gpt2_textgen')

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_input)

    if pretrained:
        print(f'>> Loading pretrained weights from Hugging Face `{model_name}` model')

        # Instantiate Hugging Face's model
        hf_model_names = {
            'gpt2': 'gpt2', 'gpt2-medium': 'gpt2-medium', 'gpt2-large': 'gpt2-large', 'gpt2-xl': 'gpt2-xl'
        }
        hf_model = TFGPT2LMHeadModel.from_pretrained(
            hf_model_names[model_name],
            from_pt=True
        )

        num_vars = len(model.trainable_variables)
        assert len(hf_model.trainable_variables) == num_vars

        for i in range(num_vars):
            weights = model.trainable_variables[i].numpy()

            hf_weights = hf_model.trainable_variables[i].numpy()
            hf_weights = np.squeeze(hf_weights)

            assert weights.shape == hf_weights.shape
            model.trainable_variables[i].assign(hf_weights)

    return model


def sample_next_token(logits, sampling_method='greedy', temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample the next token from a language model's logits using different sampling methods.
    
    Parameters:
        logits: 1D numpy array of model logits for the current step
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider (only for top_k sampling)
        top_p: float in (0,1], cumulative probability threshold for nucleus sampling (top-p)
    
    Returns:
        next_token_id: int, index of the sampled token
    """

    # Define numerically stable softmax
    def softmax(x):
        x = x.astype(np.float64)  # ensure stability for large logits
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    if sampling_method == 'greedy':
        # Take the token with the highest logit
        next_token_id = np.argmax(logits)

    elif sampling_method == 'temperature':
        # Convert logits to probabilities and sample
        probs = softmax(logits / temperature)
        next_token_id = np.random.choice(len(probs), p=probs)

    elif sampling_method == 'top_k':
        # Apply temperature scaling to logits
        logits = logits / temperature

        # Get indices and values of top-k logits
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]

        # Convert top-k logits to probabilities and sample
        top_k_probs = softmax(top_k_logits)
        next_token_id = np.random.choice(top_k_indices, p=top_k_probs)
    
    elif sampling_method == 'top_p':
        # Sort logit probabilities in descending order
        probs = softmax(logits)
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Calculate cumulative probabilities
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff index where cumulative probability exceeds top_p
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = max(1, cutoff_idx)  # ensure at least one token is included

        # Keep only top-p tokens
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = sorted_probs[:cutoff_idx]

        # Renormalize probabilities for the filtered tokens
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        next_token_id = np.random.choice(top_p_indices, p=top_p_probs)

    return int(next_token_id)


def check_next_token_sampling_params(sampling_method, temperature, top_k, top_p):
    """
    Checks that next-token sampling parameters are correctly set
    """

    if sampling_method not in ('greedy', 'temperature', 'top_k', 'top_p'):
        raise ValueError("Supported sampling methods are 'greedy', 'temperature', 'top_k', and 'top_p'")
    
    if  sampling_method in ('temperature', 'top_k', 'top_p'):
        if temperature <= 0:
            raise ValueError('temperature argument must be > 0')
        
    if sampling_method == 'top_k':
        if top_k < 1:
            raise ValueError('top-k argument must be >= 1')
        
    if sampling_method == 'top_p':
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError('top-p argument must be > 0.0 and <= 1.0')
            

def generate_text(
    model: 'tf.keras.Model', 
    prompt: str, 
    output_len: int, 
    sampling_method: str = 'greedy', 
    temperature: float = 1.0, 
    top_k: int = 1, 
    top_p: float = 1.0
) -> str: 
   
    """
    Generates an output text given an input prompt and a max number of output tokens

    Arguments:
        model: GPT-2 Keras model
        prompt: input text
        output_len: maximum number of output tokens
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider for top-k sampling
        top_p: float in (0.0, 1.0], cumulative probability threshold to use for nucleus sampling (top-p)

    Returns:
        Output text

    Sampling methods:
    ----------------
        'greedy':
            The token with the largest logit is selected (deterministic).
        'temperature':
            Logits scaled by temperature are converted to probabilities and a token
            is sampled from the distribution.
        'top_k':
            Only the top-k tokens (based on scaled logits) are considered. They are
            renormalized and a token is sampled from the distribution.
        'top_p':
            Tokens are sorted by probability (from scaled logits). The smallest set of
            tokens whose cumulative probability sum is >= top_p are kept, renormalized,
            and a token is sampled from this nucleus.
    """

    check_next_token_sampling_params(sampling_method, temperature, top_k, top_p)

    # GPT-2 context length and padding token
    context_len = 1024
    pad_token = 50256

    # Encode the prompt
    tokenizer = tiktoken.get_encoding('gpt2')
    tokens_out = tokenizer.encode(prompt)

    for _ in range(output_len):
        current_tokens = tokens_out[-context_len:]
        num_pad = context_len - len(current_tokens)

        # Pad input tokens to the left
        if num_pad > 0:
            padded_input = [pad_token] * num_pad + current_tokens
            attention_mask = [0] * num_pad + [1] * len(current_tokens)
        else:
            padded_input = current_tokens
            attention_mask = [1] * context_len

        # Run the model
        inputs = {
            'input_ids': tf.constant([padded_input], dtype=tf.int32),
            'attention_mask': tf.constant([attention_mask], dtype=tf.int32)
        }
        hidden_states = model(inputs)

        # Get the last hidden state, convert to numpy
        # and get rid of the batch dimension
        logits = hidden_states[:, -1, :]
        logits = np.squeeze(logits.numpy())

        # Sample the next token and append it to outout
        next_token = sample_next_token(
            logits,
            sampling_method=sampling_method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        tokens_out.append(int(next_token))

    # Decode the output text
    output_text = tokenizer.decode(tokens_out)

    return output_text


model = get_gpt2_model('gpt2', pretrained=True)
# model = TFGPT2LMHeadModel.from_pretrained('gpt2', from_pt=True)

# Example prompt
prompt = 'The future of artificial intelligence is'

print(f'\n>> Input text:\n{prompt}')
output_text = generate_text(
    model,
    prompt,
    output_len=50,
    sampling_method='top_p',
    temperature=1.0,
    top_p=0.9
)
print(f'\n>> Output text:\n{output_text}')

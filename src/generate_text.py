# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
import tiktoken
from utils import get_gpt2_model


def preprocess_text(text, context_len, tokenizer):
    """
    Converts a string to tokens.

    The list of tokens is:
        - truncated if it is longer than `context_len`
        - padded if it is shorter
    
    Returns:
        - Tokens and attention mask, two tensors of tf.int32 values with shape (1, context_len)
        - Attention mask values are 0 for padding tokens, 1 otherwise
    """

    tokens = tokenizer.encode(text)

    # Truncate
    tokens = tokens[:context_len]

    # Pad
    pad_token = 50256
    attention_mask = [1 for _ in range(context_len)]
    for i in range(len(tokens), context_len):
        tokens.append(pad_token)
        attention_mask[i] = 0

    tokens = tf.expand_dims(tf.constant(tokens, dtype=tf.int32), axis=0)
    attention_mask = tf.expand_dims(tf.constant(attention_mask, dtype=tf.int32), axis=0)

    return tokens, attention_mask


def generate_output_tokens(
    model, tokens_in, output_len, attention_mask=None, greedy=True, temperature=1.0, k=0
):
    """
    Generates a list of tokens of fixed length

    Arguments:
        model:
            Keras model to use to generate tokens
        tokens_in:
            Input tokens, a tensor of tf.in32 values with shape (batch_size, context_len)
        output_len:
            Length of the sequence of output tokens (positive integer)
        greedy:
            If True, the token with the largest probability is always selected as the next token.
        temperature:
            Used to scale logit values:
            - No effect if 1.0
            - Less randomness if < 1.0
            - More randomness if > 1.0
        k:
            if greater than 0:
            - The k tokens with the largest probabilities are selected
            - The next token is sampled from these k elements

    Returns:
        Sequences of tokens, a tensor of tf.int32 with shape (batch_size, context_len + output_len)
        This includes the original input tokens followed by the generated tokens.
    """

    tokens_out = tf.identity(tokens_in)
    context_len = tf.shape(tokens_in)[1]

    for _ in range(output_len):
        logits = model(tokens_in, attention_mask)

        # Only keep the logits of the last token
        logits = logits[:, -1, :]

        if greedy:
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            # temperature scaling
            logits /= temperature

            # Top-k filtering
            if k > 0:
                values, _ = tf.math.top_k(logits, k=k)
                min_values = values[:, -1, tf.newaxis]
                logits = tf.where(logits < min_values, -1e10, logits)

            # Convert to log of probabilities and sample
            next_token = tf.random.categorical(logits, num_samples=1)
            next_token = tf.squeeze(next_token, axis=-1)

        # Append new token to input tokens, truncate if needed
        tokens_in = tf.concat([tokens_in, next_token[:, None]], axis=-1)
        seq_len = tf.shape(tokens_in)[1]
        tokens_in = tf.cond(
            seq_len > context_len,
            lambda: tokens_in[:, -context_len:],
            lambda: tokens_in
        )

        tokens_out = tf.concat([tokens_out, next_token[:, None]], axis=-1)

    return tokens_out


tokenizer = tiktoken.get_encoding('gpt2')

model = get_gpt2_model('gpt2')

text = (
    'GPT-2 was created as a "direct scale-up" of GPT-1 with a ten-fold increase in both ' +
    'its parameter count and the size of its training dataset. It is a general-purpose ' +
    'learner and its ability to perform the various tasks was'
)

tokens_in, attention_mask = preprocess_text(text, context_len=1024, tokenizer=tokenizer)

tokens_out = generate_output_tokens(
    model,
    tokens_in,
    attention_mask=attention_mask,
    output_len=100,
    greedy=False,
    temperature=1.0,
    k=5
)

tokens_out = tokens_out[0]
text_out = tokenizer.decode(tokens_out.numpy().tolist())

print()
print(text_out)
print()

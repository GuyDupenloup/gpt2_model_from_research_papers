import tensorflow as tf
import tiktoken
from utils import get_gpt2_model


def preprocess_text(text, context_len, tokenizer):

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


def generate_output_tokens(model, tokens_in, output_len, greedy=True, temperature=1.0, top_k=0):

    tokens_out = tf.identity(tokens_in)
    context_len = tf.shape(tokens_in)[1]

    for _ in range(output_len):
        logits = model(tokens_in)

        # Only keep the logits of the last token
        logits = logits[:, -1, :]

        if greedy:
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            # temperature scaling
            logits /= temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = tf.math.top_k(logits, k=top_k)
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

model = get_gpt2_model('124M')

text = 'Scarlett Ingrid Johansson is an American actress and singer'

tokens_in = preprocess_text(text, context_len=1024, tokenizer=tokenizer)
tokens_out = generate_output_tokens(model, tokens_in, output_len=100, temperature=5.0)

tokens_out = tokens_in[0]
text_out = tokens_out.numpy().tolist()

print()
print(text_out)
print()

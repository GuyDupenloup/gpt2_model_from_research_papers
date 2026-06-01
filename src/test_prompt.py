# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import numpy as np
import tensorflow as tf
import tiktoken

from model_utils import create_gpt2_language_model


def sample_next_token(logits, params):
    """
    Samples the next token from a language model's logits 
    using different sampling methods.

    Arguments:
        logits:
            Model logits for the current step, 1D numpy array with shape (vocab_size,).

        params:
            Sampling parameters, a dictionary with the following items:
                "method":
                    The sampling method to use.
                    A string, one of ('greedy', 'temperature', 'top_k', 'top_p').
                "temperature":
                    Temperature scaling, a float > 0.
                "top_k":
                    Number of top-k tokens to consider ('top_k' method).
                    An integer >= 1.
                "top_p":
                    Cumulative probability threshold for nucleus sampling ('top_p' method).
                    A float in (0, 1].
            Default values:
                "method"      -> "top_k"
                "temperature" -> 0.8
                "top_k"       -> 20
                "top_p"       -> 0.9
                
    Returns:
        An integer, index in the logits of the sampled next token.
    """

    def softmax(x):
        x = x.astype(np.float64)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # Get parameter values applying defaults
    method, temperature, top_k, top_p = (
        params.get("method", "top_k"),
        params.get("temperature", 0.8),
        params.get("top_k", 20),
        params.get("top_p", 0.9),
    )

    if method == "greedy":
        next_token = np.argmax(logits)

    elif method == "temperature":
        probs = softmax(logits / temperature)
        vocab_size = logits.shape[0]
        next_token = np.random.choice(vocab_size, p=probs)

    elif method == "top_k":
        scaled_logits = logits / temperature
        
        # Select the top-k token indices (unordered)
        top_k_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
        top_k_logits = scaled_logits[top_k_indices]
        
        # Convert to probabilities and sample
        top_k_probs = softmax(top_k_logits)
        next_token = np.random.choice(top_k_indices, p=top_k_probs)

    elif method == "top_p":
        probs = softmax(logits / temperature)
        
        # Sort tokens by probability in descending order
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Find the cutoff index where cumulative probability exceeds top_p
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = max(1, cutoff_idx)  # Keep at least one token

        # Sample from the nucleus
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_probs = top_p_probs / np.sum(top_p_probs)  # Renormalize
        next_token = np.random.choice(top_p_indices, p=top_p_probs)

    return next_token


def generate_token_sequence(
    model,
    prompt_ids,
    attention_mask,
    sampling_params,
    output_len
):
    
    """
    Generates `output_len` tokens given a sequence of input tokens 
    (the prompt).

    Arguments:
        model:
            GPT-2 Keras model.

        prompt_ids:
            Input token sequence, a list of integers of length seq_len.
            
        attention_mask:
            Mask specifying which token positions to attend to.
            A list of length 1024.
            
        sampling_params:
            Sampling parameters, a dictionary.
            See docstring of function sample_next_token().

        output_len:
            Length of full token sequence (prompt + model's response)

    Returns:
        Full token sequence (prompt + generated tokens)
        A numpy array with shape (seq_len,)
    """

    tokens_out = np.array(prompt_ids, dtype=np.int32)
    masks_out  = np.array(attention_mask, dtype=np.int32)

    for _ in range(output_len):

        hidden_states = model({
            "input_ids": tf.constant([tokens_out], dtype=tf.int32),
            "attention_mask": tf.constant([masks_out], dtype=tf.int32)
        }).numpy()

        # Get the index of the last token before padding
        last_token_index = masks_out.sum() - 1

        logits = hidden_states[0, last_token_index, :]  # (vocab_size,)

        next_token = sample_next_token(logits, sampling_params)

        pad_start = masks_out.sum()
        tokens_out[pad_start] = next_token
        masks_out[pad_start] = 1

    return tokens_out


def test_prompt(model_size, prompt, sampling_params, output_len):

    print(f"\n>> Creating GPT-2 model `{model_size}`")
    model = create_gpt2_language_model(model_size)

    tokenizer = tiktoken.get_encoding("gpt2")
    seq_len = 1024
    pad_token = 50256

    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = prompt_ids[:seq_len]

    attention_mask = [1] * len(prompt_ids)

    if len(prompt_ids) < seq_len:
        pad_len = seq_len - len(prompt_ids)
        prompt_ids += [pad_token] * pad_len
        attention_mask += [0] * pad_len

    print(f"\n>> Prompt: {prompt}")
    
    raw_tokens_out = generate_token_sequence(
        model,
        prompt_ids,
        attention_mask,
        sampling_params,
        output_len
    )

    tokens_out = []
    for tk in raw_tokens_out:
        if tk != pad_token:
            tokens_out.append(tk)

    text_out = tokenizer.decode(tokens_out)

    print(f"\n>> Model output: {text_out}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--project_root",
        help="Directory where to save the dataset files (metadata, TFRecords)",
        type=str,
        default=None
    )
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_size",
        help="GPT-2 model size, one of ('124M', '355M', '774M', '1542M')",
        type=str,
        default="124M"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt",
        type=str,
    )
    parser.add_argument(
        "--sampling_params",
        help="Sampling parameters",
        type=str,
        default="{'method': 'top_k', 'temperature': 0.8, 'top_k': 20}"
    )
    parser.add_argument(
        "--output_len",
        help="Model output length",
        type=int,
        default=100
    )

    args = parser.parse_args()

    try:
        sampling_params = eval(args.sampling_params)
    except:
        raise ValueError("Argument `sampling_params` must be a valid dictionary")
    
    test_prompt(args.model_size, args.prompt, sampling_params, args.output_len)
 
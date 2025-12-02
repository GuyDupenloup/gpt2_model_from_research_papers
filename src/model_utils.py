
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from tabulate import tabulate
import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from gpt2_model import GPT2LanguageModel


def get_gpt2_model_config(model_size):
    """
    Provides model parameters for OpenAI's model sizes.
    """
    model_configs = {
         '124M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
         '355M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
         '774M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
        '1.56B': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
    }
    assert model_size in model_configs
    config = model_configs[model_size]
    config['size'] = model_size

    return config


def create_gpt2_language_model(model_size):
    
    model_config = get_gpt2_model_config(model_size)
    model = GPT2LanguageModel(model_config, dropout_rate=0.1, name='gpt2_LM')

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    _ = model(dummy_input)

    return model


def print_trainable_variables(model):
    """
    Prints the trainable variables of a model (name, shape, number of parameters)
    """

    print('\n' + '=' * 80)
    print(f"  Trainable variables of model `{model.config['size']}`")
    print('=' * 80 + '\n')

    headers = ['Variable', 'Shape', '#Params']
    data = []
    total_params = 0

    for var in model.trainable_variables:
        num_params = int(np.prod(var.shape))
        total_params += num_params
        data.append([var.name, var.shape, f'{num_params:,.0f}'])

    print(tabulate(data, headers=headers, tablefmt='pipe', colalign=('left', 'center', 'right')))
    print(f'\nTotal trainable parameters: {total_params:,.0f}')


def transfer_pretrained_weights(model, verbose=True):
    """
    Creates a pretrained Hugging Face model of the requested size
    and returns its trainable variables.
    """

    model_size = model.config['size']

    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

	# Check that the two models have the same 
	# number of trainable variables
    num_vars = len(hf_model.trainable_variables) 
    assert len(model.trainable_variables) == num_vars

    if verbose:
        print('\n' + '=' * 80)
        print(f'  Trainable variables of model `{model_size}` - Hugging Face in 2nd position')
        print('=' * 80)

    for i in range(num_vars):
        var = model.trainable_variables[i]
        weights = var.numpy()

        hf_var = hf_model.trainable_variables[i]
        hf_weights = hf_var.numpy()

        if verbose:
             print(f'\nvar #{i}')
             print(var.name, '   ', var.shape)
             print(hf_var.name, '   ', hf_var.shape)

        # Convert shapes (1, N) to (N,)
        hf_weights = np.squeeze(hf_weights)

        assert weights.shape == hf_weights.shape
        var.assign(hf_weights)


model = create_gpt2_language_model('124M')
print_trainable_variables(model)
transfer_pretrained_weights(model, verbose=True)

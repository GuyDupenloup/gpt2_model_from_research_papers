# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from tabulate import tabulate
import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from gpt2_model import GPT2TextGenModel


def get_gpt2_model_config(model_name):
    gpt2_configs = {
        '117M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
        '345M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
        '774M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
        '1542M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
    }

    assert model_name in gpt2_configs
    config = gpt2_configs[model_name]
    config['name'] = model_name

    return config


def get_hf_model_name(model_name):
    name_mapping = {
        '117M': 'gpt2', '345M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'
    }
    assert model_name in name_mapping
    
    return name_mapping[model_name]


def get_gpt2_model(model_name, pretrained=True):

    # Create the model
    model_config = get_gpt2_model_config(model_name)
    model = GPT2TextGenModel(model_config)

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_inputs = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_inputs)

	if pretrained:
		# Get corresponding HF model name
		hf_model_name = get_hf_model_name(model_name)
		print(f'>> Loading pretrained weights from Hugging Face `{hf_model_name}` model')

		# Get HF model with pretrained weights from PyTorch
		hf_model = TFGPT2LMHeadModel.from_pretrained(hf_model_name, from_pt=True)

		# Check that the two models have the same number of trainable variables
		num_vars = len(model.trainable_variables)
		hf_num_vars = len(hf_model.trainable_variables)
		assert num_vars == hf_num_vars

		for i in range(num_vars):
			weights = model.trainable_variables[i].numpy()
			hf_weights = hf_model.trainable_variables[i].numpy()

			# Reshape (1, N) weight shapes from HF's model to (N,)
			hf_weights = np.squeeze(hf_weights)

			# Check that the weights have the same shape and transfer them
			assert weights.shape == hf_weights.shape
			model.trainable_variables[i].assign(hf_weights)

    return model


def print_trainable_vars(model_name):
    """
    Prints model's trainable variables (name, shape, number of parameters)
    """

    model = create_gpt2_model(model_name, pretrained=False)

    print('\n' + '=' * 80)
    print(f"  Trainable variables of model `{model_name}`")
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


def compare_trainable_vars(model_name):
    """
    Prints trainable variables (names, shapes, number
    of parameters) for all GPT-2 model sizes
    """

    # Create two equivalent models
    model = create_gpt2_model(model_name, name='my_model')

    hf_model_name = get_hf_model_name(model_name)
    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_model_name, from_pt=True, name='HF_model')
    
    print('\n' + '=' * 80)
    print(f'  Trainable variables compared')
    print('=' * 80 + '\n')

    num_vars = len(model.trainable_variables)
	assert len(hf_model.trainable_variables) == num_vars
	
    for i in num_vars:
		var = model.trainable_variables[i]
		hf_var = hf_model.trainable_variables[i]
		print(f'\n{var.name}    {var.shape}')
		print(f'{hf_var.name}    {hf_var.shape}')

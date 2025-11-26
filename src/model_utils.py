

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
    mapping = {
        '117M': 'gpt2',
        '345M': 'gpt2-medium',
        '774M': 'gpt2-large',
        '1542M': 'gpt2-xl'
    }
    assert model_name in  mapping
    return mapping[model_name]


def get_gpt2_model(model_name, pretrained=True):

    print(f'Creating `{model_name}` model')

    model_config = get_gpt2_model_config(model_name)
    model = GPT2TextGenModel(model_config, dropout_rate=0.1, name='gpt2_textgen_model')

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_input)

	if pretrained:
		# Load corresponding Hugging Face model
		hf_model_name = get_hf_model_name(model_name)
		print(f'Loading pretrained weights from Hugging Face model `{hf_model_name}`')
		hf_model = TFGPT2LMHeadModel.from_pretrained(hf_model_name, from_pt=True)

		# Check that the two models have the same 
		# number of trainable variables
		num_vars = len(model.trainable_variables)
		assert len(hf_model.trainable_variables) == num_vars

		for i in range(num_vars):
			var = model.trainable_variables[i]
			weights = var.numpy()

			hf_var = hf_model.trainable_variables[i]
			# Convert weight shapes (1, N) to (N,)
			hf_weights = np.squeeze(hf_var.numpy())

			# Check that the weight shapes match and copy weights
			assert weights.shape == hf_weights.shape
			var.assign(hf_weights)

    return model


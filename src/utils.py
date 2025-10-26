
import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from gpt2_model import GPT2Model


MODEL_CONFIG = {
    '124M':  {'vocab_size': 50257,  'context_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
    '355M':  {'vocab_size': 50257,  'context_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
    '774M':  {'vocab_size': 50257,  'context_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
    '1542M': {'vocab_size': 50257,  'context_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25},
}


def load_pretrained_weights(model, model_size):

    print(f'>>> Loading pretrained weights')

    # Get Hugging Face's model
    sizes = {'124M': 'gpt2', '355M': 'medium', '774M': 'large', '1542M': 'XL'}
    hf_model = TFGPT2LMHeadModel.from_pretrained(sizes[model_size], from_pt=True)

    # Get the pretrained weights
    weights = hf_model.get_weights()

    for i in range(len(weights)):
        ws = np.shape(weights[i])
        if len(ws) == 2 and ws[0] == 1:
            weights[i] = np.reshape(weights[i], (ws[1],) )

    model.set_weights(weights)


def get_gpt2_model(model_size, pretrained=False):
    
    if model_size not in MODEL_CONFIG:
        raise ValueError(f'Valid model sizes are {list(MODEL_CONFIG.keys())}. Received {model_size}')

    params = MODEL_CONFIG[model_size]

    print(f'>>> Creating {model_size} model')
    model = GPT2Model(
        vocab_size=params['vocab_size'],
        seq_len=params['context_len'],
        d_model=params['d_model'],
        n_heads=params['n_heads'],
        n_layers=params['n_layers'],
        name=model_size
    )

    # Use dummy data to build the model
    dummy_input = tf.random.uniform(
        (1, params['context_len']),
        minval=0,
        maxval=params['vocab_size'],
        dtype=tf.int32
    )
    _ = model(dummy_input)

    if pretrained:
        load_pretrained_weights(model, model_size)

    return model


def print_trainable_variables(model):

    total_params = 0
    print("\n=== All Trainable Variables ===\n")
    for var in model.trainable_variables:
        num_params = np.prod(var.shape)
        total_params += num_params
        print(f'{var.name}: {var.shape} || num_params: {num_params}')

    print(f'Total trainable parameters: {total_params}')

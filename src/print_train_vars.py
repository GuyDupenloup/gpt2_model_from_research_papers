import numpy as np
import tensorflow as tf

from utils import MODEL_CONFIG
from gpt2_model_prelim import GPT2Model


def print_trainable_variables(model):

    total_params = 0
    for var in model.trainable_variables:
        num_params = np.prod(var.shape)
        total_params += num_params
        print(f'{var.name}:   {var.shape} = {num_params}')

    print(f'Total trainable parameters: {total_params}')


for name, params in MODEL_CONFIG.items():

    print(f'>>> Creating model {name}')
    model = GPT2Model(
        vocab_size=params['vocab_size'],
        seq_len=params['context_len'],
        d_model=params['d_model'],
        n_heads=params['n_heads'],
        n_layers=params['n_layers']
    )

    # Build the model using dummy data
    dummy_input = tf.random.uniform(
        (1, params['context_len']),
        minval=0,
        maxval=params['vocab_size'],
        dtype=tf.int32
    )
    _ = model(dummy_input)
    
    print('>>> Printing trainable variables')
    print_trainable_variables(model)

    exit()

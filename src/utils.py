
from tabulate import tabulate
import numpy as np
import tensorflow as tf

from transformers import TFGPT2LMHeadModel
from gpt2_model import GPT2Model


VOCAB_SIZE = 50257
CONTEXT_LEN = 1024

MODEL_CONFIGS = {
    '124M':  {'d_model': 768,  'n_layers': 12, 'n_heads': 12},
    '355M':  {'d_model': 1024, 'n_layers': 24, 'n_heads': 16},
    '774M':  {'d_model': 1280, 'n_layers': 36, 'n_heads': 20},
    '1542M': {'d_model': 1600, 'n_layers': 48, 'n_heads': 25},
}


def load_pretrained_weights_(model, model_size):

    print(f'>>> Loading pretrained weights')

    # Get Hugging Face's model
    hf_sizes = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_sizes[model_size], from_pt=True)

    # Get the pretrained weights
    weights = hf_model.get_weights()

    for i in range(len(weights)):
        ws = np.shape(weights[i])
        if len(ws) == 2 and ws[0] == 1:
            weights[i] = np.squeeze(weights[i])

    model.set_weights(weights)


def get_gpt2_model(model_size, pretrained=False):
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f'Valid model sizes are {list(MODEL_CONFIGS.keys())}. Received {model_size}')

    params = MODEL_CONFIGS[model_size]

    print(f'>>> Creating {model_size} model')
    model = GPT2Model(
        vocab_size=VOCAB_SIZE,
        seq_len=CONTEXT_LEN,
        d_model=params['d_model'],
        n_heads=params['n_heads'],
        n_layers=params['n_layers'],
        name=model_size
    )

    # Use dummy data to build the model
    dummy_input = tf.random.uniform((1, CONTEXT_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
    _ = model(dummy_input)

    if pretrained:
        load_pretrained_weights_(model, model_size)

    return model


def model_summary(model, name):

    print(f'\n====== Trainable variables in model {name} ======n')

    headers = ['Variable', 'Shape', '#Params']
    data = []
    total_params = 0

    for var in model.trainable_variables:
        num_params = int(np.prod(var.shape))
        total_params += num_params
        data.append([var.name, var.shape, f'{num_params:,.0f}'])

    print(tabulate(data, headers=headers, tablefmt='pipe', colalign=('left', 'center', 'right')))
    print(f'\nTotal trainable parameters: {total_params:,.0f}')


def all_models_summary():

    for name, params in MODEL_CONFIGS.items():

        print(f'>>> Creating model {name}')
        model = GPT2Model(
            vocab_size=VOCAB_SIZE,
            seq_len=CONTEXT_LEN,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers']
        )
          
        # Use dummy data to build the model
        dummy_input = tf.random.uniform((1, CONTEXT_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        _ = model(dummy_input)

        model_summary(model, name)

        exit()


def all_hf_models_summary():

    for name in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'):

        print(f'>>> Creating model {name}')
        model = TFGPT2LMHeadModel.from_pretrained(name, from_pt=True)

        # Use dummy data to build the model
        dummy_input = tf.random.uniform((1, CONTEXT_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        _ = model(dummy_input)
     
        model_summary(model, name) 

        exit()

all_models_summary()

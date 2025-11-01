
from tabulate import tabulate
import numpy as np
import tensorflow as tf

from transformers import TFGPT2LMHeadModel
from gpt2_model_aligned import GPT2Model, MODEL_CONFIGS


def get_gpt2_model(model_size, pretrained=True):
    """
    Creates a GPT-2 model, and loads the pretrained weights from
    the corresponding Hugging Face model if `pretrained` is True.
    """

    # Get the model and build it
    config = MODEL_CONFIGS[model_size]
    model = GPT2Model(model_size)

    # Build the model
    dummy_input = tf.random.uniform((1, config['seq_len']), minval=0, maxval=config['vocab_size'], dtype=tf.int32)
    _ = model(dummy_input)

    if pretrained:
        print(f'>>> Loading pretrained weights from Hugging Face `{model_size}` model')

        # Get Hugging Face's model
        hf_model = TFGPT2LMHeadModel.from_pretrained(model_size, from_pt=True)

        # Get the pretrained weights
        weights = hf_model.get_weights()

        # Convert shapes (1, N) to (N,)
        vars = model.trainable_variables

        for i in range(len(weights)):
            var_name = vars[i].name
            ws = np.shape(weights[i])
            if var_name[-6:] == 'bias:0' and len(ws) == 2 and ws[0] == 1:
                weights[i] = np.squeeze(weights[i])

        model.set_weights(weights)

    return model


def model_summary(model):
    """
    Prints model's trainable variables (names, shapes, number of parameters)
    """

    print('\n' + '=' * 80)
    print(f"  Trainable variables of model `{model.name}`")
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


def compare_train_vars(model_size):
    """
    Prints trainable variables (names, shapes, number
    of parameters) for all GPT-2 model sizes
    """

    # Get the models
    model = get_gpt2_model(model_size)
    hf_model = TFGPT2LMHeadModel.from_pretrained(model_size, from_pt=True)
    
    print('\n' + '=' * 80)
    print(f"  Trainable variables of `{model_size}` models")
    print("Hugging Face's model variables are in second position")
    print('=' * 80 + '\n')

    vars = model.trainable_variables
    hf_vars = hf_model.trainable_variables
    if len(vars) != len(hf_vars):
        raise ValueError(
            f"The models don't have the same number of trainable variables ({len(vars)} vs. {len(hf_vars)})"
        )

    for i in range(len(model.trainable_variables)):
        print(f'i = {i}')
        print(f'{vars[i].name}    {vars[i].shape}')
        print(f'{hf_vars[i].name}    {hf_vars[i].shape}')
        print()

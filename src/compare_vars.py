# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from model_utils import create_gpt2_language_model
from transformers import TFGPT2LMHeadModel


def compare_model_variables(model_size):

    # Get GPT2 model
    model = create_gpt2_language_model(model_size)

    # Get equivalent Hugging Face model
    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    print('\n' + '=' * 80)
    print(f'Trainable variables of model `{model_size}` - Hugging Face in 2nd position)')
    print('=' * 80)

    # Check that the two models have the same number of trainable variables
    assert len(model.trainable_variables) == len(hf_model.trainable_variables)

    for i in range(len(model.trainable_variables)):

        var = model.trainable_variables[i]
        hf_var = hf_model.trainable_variables[i]

        print(f'\nvar #{i}')
        print(f'{var.name}    {var.shape}')
        print(f'{hf_var.name}    {hf_var.shape}')


compare_model_variables('124M')

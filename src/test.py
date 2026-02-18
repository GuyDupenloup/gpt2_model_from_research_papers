
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from model_utils import create_gpt2_language_model, print_trainable_variables, transfer_pretrained_weights

model = create_gpt2_language_model('124M')
transfer_pretrained_weights(model, verbose=True)
print_trainable_variables(model)

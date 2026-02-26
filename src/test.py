
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from model_utils import create_gpt2_language_model, print_model_variables
from gen_text import generate_text

model = create_gpt2_language_model('124M')
print_model_variables(model)

# Example prompt
prompt = 'The secret to live a happy life is'
print(f'\n>> Prompt:\n{prompt}')

output_text = generate_text(
    model,
    prompt,
    output_len=50,
    sampling_method='top_p',
    temperature=0.8,
    top_k=20,
    top_p=0.9
    
)
print(f'\n>> Output text:\n{output_text}')

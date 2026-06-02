# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import argparse
from model_utils import create_gpt2_language_model
from transformers import TFGPT2LMHeadModel


def print_variables_side_by_side(model_size):

    """
    Creates two GPT-2 models of the same size: a GPT2LanguageModel model
    from gpt2_language_model.py and another one from the transformers
    package developed by Hugging Face.
    Then, loops through the trainable variables of the two models and
    prints the shape of the weights assigned to them.

    Arguments:
        model_size: Size of the models, one of ("124M", "355M", "774M", "1542M").

    Trainable variables of the two models are printed as shown in the example below:

        var #1
        gpt2_LM/gpt2_LM/pos_emb/embeddings:0     (1024, 768)
        tfgpt2lm_head_model_3/transformer/wpe/embeddings:0     (1024, 768)

    In first position is the GPT2LanguageModel variable, and in second position
    is the corresponding Hugging Face variable. Variable names are different 
    because different names were used in the two models. What matters is that
    the weight shapes match.
    """

    # Get GPT2 model
    model = create_gpt2_language_model(model_size)

    # Get equivalent Hugging Face model
    mapping = {"124M": "gpt2", "355M": "gpt2-medium", "774M": "gpt2-large", "1542M": "gpt2-xl"}
    assert model_size in mapping
    hf_name = mapping[model_size]

    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    print("\n" + "=" * 80)
    print(f"Trainable variables for model size`{model_size}` - Hugging Face in 2nd position")
    print("=" * 80)

    # Check that the two models have the same number of trainable variables
    assert len(model.trainable_variables) == len(hf_model.trainable_variables)

    for i in range(len(model.trainable_variables)):

        var = model.trainable_variables[i]
        hf_var = hf_model.trainable_variables[i]

        print(f"\nvar {i}")
        print(f"{var.name}    {var.shape}")
        print(f"{hf_var.name}    {hf_var.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    parser.add_argument(
        "--model_size",
        help="GPT-2 model size, one of ('124M', '355M', '774M', '1542M')",
        type=str,
        default="124M"
    )
    args = parser.parse_args()

    print_variables_side_by_side(args.model_size)
 
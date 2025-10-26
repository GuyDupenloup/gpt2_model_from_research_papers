import numpy as np
import tensorflow as tf

from transformers import TFGPT2LMHeadModel, GPT2Config
from final_model import GPT2Model


config = GPT2Config(
    vocab_size=50257,      # Size of tokenizer vocabulary
    n_positions=1024,      # Maximum sequence length
    n_ctx=1024,            # Context size (should match n_positions)
    n_embd=768,            # Hidden size (d_model)
    n_layer=2,            # Number of transformer layers
    n_head=12,             # Number of attention heads
    resid_pdrop=0.1,       # Dropout for residuals
    embd_pdrop=0.1,        # Dropout for embeddings
    attn_pdrop=0.1,        # Dropout for attention weights
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256
)

# Create and build the smallest Hugging Face model
hf_model = TFGPT2LMHeadModel(config)

dummy_input = tf.random.uniform((1, 1024), minval=0, maxval=50257, dtype=tf.int32)
_ = hf_model(dummy_input)


# Create and build my model
my_model = GPT2Model(
    vocab_size=50257,
    max_seq_len=1024,
    d_model=768,
    n_heads=12,
    n_layers=2
)

dummy_input = tf.random.uniform((1, 1024), minval=0, maxval=50257, dtype=tf.int32)
_ = my_model(dummy_input)


# Print variables in the Hugging Face model
total_params = 0
print("\n=== All Trainable Variables in Hugging Face's model ===\n")
for var in hf_model.trainable_variables:
    var_shape = var.shape
    num_params = np.prod(var.shape)
    total_params += num_params
    print(f"{var.name}: {var_shape} = {num_params:,} params")

print(f"\nTotal: {total_params:,} parameters")


# Print variables in my model
total_params = 0
print("\n=== All Trainable Variables in Hugging Face's model ===\n")
for var in my_model.trainable_variables:
    num_params = np.prod(var.shape)
    total_params += num_params
    print(f"{var.name}: {var.shape} = {num_params:,} params")

print(f"\nTotal: {total_params:,} parameters")

weights = hf_model.get_weights()

print("======== Weights shapes ======")
for i in range(len(weights)):
    ws = np.shape(weights[i])
    if len(ws) == 2 and ws[0] == 1:
        weights[i] = np.reshape(weights[i], (ws[1],) )

my_model.set_weights(weights)

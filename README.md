
# GPT-2 Model in Tensorflow

## 1. Introduction

This repo contains a Tensorflow implementation of OpenAI's GPT-2 model.

The goal of this project was to construct the model using only these three research papers:

- The original transformer paper published in 2017:
    [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
    "Attention Is All You Need".
    https://arxiv.org/abs/1706.03762

- The GPT paper published by OpenAI in 2018:
    [2] Alec Radford, Karthik Narasimhan, Tim Salimar, Ilya Sutskever.
    "Improving Language Understanding by Generative Pre-Training"
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

- The GPT-2 paper published by OpenAI in 2019:
    [3] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.
    Language Models are Unsupervised Multitask Learners".
     https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com

We refer to these papers as Transformer Paper, GPT Paper, and GPT-2 Paper.


## 2. GPT-2 model architecture

### Original transformer architecture

The transformer architecture from the original Transformer Paper is reproduced below:

![](pictures/transformer_architecture.JPG)
![](pictures/self_attention.JPG)

encoder + decoder

Layers, hidden size.

### Decoder-only architecture

In the section entitled "Model Specifications" of the GPT Paper, the authors state:

    "Our model largely follows the original transformer work. We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads).

And in section "2.3 Model" of the GPT-2 Paper:

    "We use a Transformer (Vaswani et al., 2017) based architecture for our LMs. The model largely follows the details of the OpenAI GPT model (Radford et al., 2018) with a few modifications."


The figure from the GPT-1 paper is shown below:

![](pictures/GPT_architecture.JPG)

As we can see in the figure, the GPT model is the right-end part of the Transformer Paper diagram above, which is the decoder.

In the original Transformer Decoder, the layer is built with 3 sub-layers:
- Masked multi-head self-attention + layer norm
- Encoder-decoder cross-attention + layer norm
- Feed-forward network + layer norm

Because it is a decoder-only architecture, the GPT-2 model only needs 2 sub-layers:
- Masked multi-head self-attention + layer norm
- Feed-forward network + layer norm

### Layer normalization

In section "2.3 Model" of the GPT-2 Paper:

    "Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an additional layer normalization was added after the final self-attention block."

### Dropout layers

In Section "5.4 Regularization" of the Transformer Paper:

"We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop=0.1."

As dropout layers are not mentioned in the GPT and GPT2 Papers, we assume that they are the same as the Transformer Paper.

### GPT-2 architecture diagram

We can now draw the architecture of the complete GPT-2 model.

![](pictures/GPT2_architecture.JPG)


### Head output size and number of heads

From GPT-2 paper, "section 2.3 Model":

    "The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batch size of 512 is used."

Table 1 in GPT-2 Paper:

| Parameters | Layers |  d_model  |
|------------|--------|-----------|
| 117M       |   12   |      768  |
| 345M       |   24   |     1024  |
| 762M       |   36   |     1280  |
| 1542M      |   48   |     1600  |

The table does not provide the size of the head output and the number of heads in parallel in the multi-head attention block.

In section "3.2.2 Multi-Head Attention" of the Transformer Paper, the authors specify:

“We employ h=8 parallel attention layers, or heads. For all models, we use d_model=512 and d<sub>k</sub> = d<sub>v</sub> = d_model/h = 64"

d<sub>k</sub> = d<sub>v</sub> is the size of the Key and Value matrices. Although not mentioned, d<sub>q</sub> is equal to d<sub>k</sub> and d<sub>v</sub>. We will use d_head for the size of these 3 matrices.

==> Same for the projection matrix.

So the paper gives the following relationship:

        n_heads = d_model / d_head

With d_head=64

The GPT and GPT-2 Papers don't indicate any change in the value of d_head, so we assume that it is 64 for the GPT-2 model sizes.

We can now add the number of heads n_heads to the table.

| Parameters | Layers |  d_model  |  d_head  | n_heads  |
|------------|--------|-----------|----------|----------|
| 117M       |   12   |      768  |    64    |    12    |
| 345M       |   24   |     1024  |    64    |    16    |
| 762M       |   36   |     1280  |    64    |    20    |
| 1542M      |   48   |     1600  |    64    |    25    |


### K/Q/V matrices

In section "3.2.1 Scaled Dot-Product Attention" of the Transformer Paper:
    "In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V."

Thus, the K, Q and V matrices are packed in a single matrix rather than having 3 distinct matrices. This makes the computation of products by the input matrices more efficient.

### Feed-forward network

A feed-forward network is applied to the entire d_model dimension after the multi-head attention has combined all heads back together.

Section 3.3 "Position-wise Feed-Forward Networks" of the Transformer Paper states:

    "In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically This consists of two linear transformations with a ReLU activation in between."

Further in the same section:

    "The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality dff = 2048."

The network has 2 layers, and the size of the inner layer is 4x the size of the output layer.

In the GPT paper, "Model specifications" section:

    "We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states."

As in the original Transformer Paper, the size of the inner layer is 4x the size of the output layer. As there is no mention of it in the GPT-2 Paper, we assume that this 4x ratio is valid for all the GPT-2 model sizes.

Moreover, in the "Model Specifications" section of the GPT Paper, it is indicated that the ReLU activation function of the original transformer was replaced by a GELU:

    "For the activation function, we used the Gaussian Error Linear Unit (GELU)."


### Output linear layer

The Transformer Paper describes the model output linear layer in section "3.4: Embeddings and Softmax":

"Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by sqrt(d_model)."

The GPT Paper describes the same linear layer in section "3.2 Supervised fine-tuning". The GPT-2 paper does not mention it, so we assume that it is the same as in the Transformer Paper.

From the papers, we can infer that the output linear layer projects the final decoder representations to the vocabulary size, followed by a softmax function to produce token probabilities. The weights of the projection matrix Wy are shared with the embedding matrix E. The input embedding layer maps tokens to embeddings, while the output linear transformation maps embeddings back to tokens. Therefore, Wy is the transpose of E.

Although not specified, the weight sharing between E and Wy is the token embeddings only. Positional encodings are part of the attention mechanism, not the token/embedding conversion.


## 3. First model implementation

The first model that I wrote, using the three research papers only, is in file **src/gpt2_model_prelim.py**

==> Source code files

Using the **src/parameter_count.py** script, I counted the number of parameters for each model size and obtained the following numbers.


| GPT-2 Paper   | My model    |
|---------------|-------------|
| 117M          |   124M      |
| 345M          |   355M      |
| 762M          |   774M      |
| 1542M         |  1542M      |

The first two model sizes are different from the numbers that are given in the GPT-2 Paper. As I could not find any explanation, I finally did some research and found out from different sources that my numbers are actually correct.

## 4. Aligning with OpenAI's model

I got the OpenAI model using the **transformers** package developed by Hugging Face (HF).

To compare the architecture of HF's model with my model, I printed the trainable variables of the two models using the **print_training_variables()** from the **utils.py** file. The result is in file **trainable_variables.txt**.

Like my model, HF's model strictly follows the architecture described in the research papers. Although names are different, the two architectures are identical expect for three differences:
1. My model uses 3 separate matrices for Wq, Wk and Wv in the attention head, while they are concatenated in a single matrix in HF's model.
2. My model has a softmax at the output, while HF's model outputs logits.
3. The shape of biases in dense layers is (1, N) in HF's model, while it is (N,) in my model.

Differences #1 and #2 were trial to fix. I aligned on HF's model.

Difference #3 has to do with broadcasting. Take for example the Keras layer used for the output projection of the attention heads:

    Wv = tf.keras.layers.Dense(d_model)

Omitting the batch dimension, the layer input X has shape (seq_len, d_model). The layer performs:

    X * Wv + b

As X * Wv has shape (seq_len, d_model), the bias b must be broadcasted across the seq_len dimension to shape (seq_len, d_model).

Keras automatically takes care of the broadcasting. By using a bias with shape (1, N), HF's model makes the broadcasting axis explicit. For the sake of simplicity, I kept the standard Keras layers and avoided the introduction of a custom layer with bias (1, N).

## 5. Loading pretrained weights

## 6. Attention mask

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

In Section "3.3 Input Representation"

“To limit the impact of padding tokens, we apply an attention mask to avoid performing attention on padding positions.”

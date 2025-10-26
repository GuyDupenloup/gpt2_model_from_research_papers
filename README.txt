
# ChatGPT-2 Model in Tensorflow

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


## 2. Model architecture

In the section entitled "Model Specifications" of the GPT paper, the authors state:

    "Our model largely follows the original transformer work."

And in section "2.3 Model" of the GPT-2 paper:

    "We use a Transformer (Vaswani et al., 2017) based architecture for our LMs. The model largely follows 
    the details of the OpenAI GPT model (Radford et al., 2018) with a few modifications.

Therefore, when the information we need is missing in the GPT-2 paper, we can first refer to: 1) the GPT paper
2) the tranformer paper.

From the GPT paper "Model Specifications" section:

"Our model largely follows the original transformer work [62]. We trained a 12-layer decoder-only transformer 
with masked self-attention heads (768 dimensional states and 12 attention heads)."

Referring to the transformer paper, we can deduct that the GPT and GPT-2 architectures look like that:






The authors of the GPT paper states

The authors of the GPT-2 table provides this table (Table 1 in the paper):

Parameters   Layers    d_model
------------------------------
117M           12       768
345M           24       1024
762M           36       1280
1542M          48       1600

In this table, the "Layers" column gives the number of transformer blocks, and the "d_model" column the size
of the embeddings.

They don't provide the 

 There are four flavors of the GPT-2 

The authors of the GPT paper [2] states in the "Model Specifications" section:
"Our model largely follows the original transformer work [1]. We trained a 12-layer decoder-only transformer
with masked self-attention heads (768 dimensional states and 12 attention heads). 
For the position-wise feed-forward networks, we used 3072 dimensional inner states. We used the Adam optimization
scheme [27] with a max learning rate of 2.5e-4. 
The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a 
cosine schedule. Wetrain for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens. 
Since layernorm [2] is used extensively throughout the model, a simple weight initialization of N(0002)was 
sufficient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53] and residual, embedding, 
and attention dropouts with a rate of 0.1 for regularization. We also employed a modified version 
of L2 regularization proposed in [37], with w = 001 on all non bias or gain weights. For the activation function, 
we used the Gaussian Error Linear Unit (GELU) [18]. We used learned position embeddings instead of the sinusoidal 
version proposed in the original work. We use the ftfy library2 to clean the raw text in BooksCorpus, standardize 
some punctuation and whitespace, and use the spaCy tokenizer.



OpenAI papers:
------------- 
GPT:
Improving Language Understanding by Generative Pre-Training
Alec Radford, Karthik Narasimhan, Tim Salimar, Ilya Sutskever
2018
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

GPT2:
Language Models are Unsupervised Multitask Learners
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
2019
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com

From GPT paper:
    - architecture of the model in figure 1
    - 12-layer decoder-only transformer with masked self-attention heads 
      (768 dimensional states and 12 attention heads).
    - contiguous sequences of 512 tokens

From GPT2 paper:
    - The vocabulary size 50,257
    - The context size from 512 to 1024 tokens 
    - Layer normalization was moved to the input of each sub-block.
    - An additional layer normalization was added after the final self-attention block.
    - A modified initialization which accounts for the accumulation on the residual path with model depth is used. 
    - We scale the weights of residual layers at initialization by a factor of 1 / sqrt(N) where N is the number 
      of residual layers. 

Parameters   Layers    d_model
------------------------------
117M           12       768
345M           24       1024
762M           36       1280
1542M          48       1600

d_model is the embedding size.
    That’s because the input token embeddings and all layer hidden states share the same dimensionality, 
    so that residual connections can be applied without projection.


The paper doesn’t list n_head explicitly, but in all GPT models the design choice is:
    Each attention head operates over a subspace of size d_head = d_model / n_head

The authors keep d_head = 64 for all models (a common and efficient choice also used in the original 
Transformer and later GPT variants). So:

    n_head = d_model / d_head = d_model / 64

This gives:

Parameters  Layers  d_model       d_head       n_head
------------------------------------------------
117M            768           64          12
345M           1024           64          16
762M           1280           64          20
1542M          1600           64          25

Where d_head = 64 actually comes from?

- The original Transformer paper (Vaswani et al., 2017), 
  where d_model = 512 and n_head = 8, giving d_head = 64,
  and OpenAI’s released GPT-2 code and model configuration files.

- Confirmed by the public GPT-2 config files in OpenAI’s GitHub repository
  and in Hugging Face’s implementation

n_layer is the number of transformers.

For the output FFN:
- GPT-1 paper:
“We use a modified Transformer decoder with the same hyperparameters 
as in Vaswani et al. (2017), except for using Gaussian Error Linear Units (GELUs) instead of ReLUs.”
- Described in attention is all you need paper
- From GPT2 paper, “Our model largely follows the GPT architecture from Radford et al. (2018):
and that it is a decoder-only Transformer. Therefore, the same FFN applies.

Head output size
---------------------------------------
🧩 1. From the original Transformer (Vaswani et al., 2017)

In Attention Is All You Need, Section 3.2.2 (“Multi-Head Attention”):

“We employ 
ℎ
=
8
h=8 parallel attention layers, or heads.
For each of these we use dimensions of 
𝑑
𝑘
=
𝑑
𝑣
=
𝑑
𝑚
𝑜
𝑑
𝑒
𝑙
/
ℎ
=
64.
d
k
	​

=d
v
	​

=d
model
	​

/h=64.”

So in the base Transformer:

𝑑
𝑚
𝑜
𝑑
𝑒
𝑙
=
512
d
model
	​

=512

ℎ
=
8
h=8

⇒ 
𝑑
ℎ
𝑒
𝑎
𝑑
=
512
/
8
=
64
d
head
	​

=512/8=64

Each head projects to a 64-dimensional query/key/value subspace.

That’s where the 64 comes from originally.

🚀 2. From GPT-1 (2018)

The GPT-1 paper (Improving Language Understanding by Generative Pre-Training) says:

“Our model is a 12-layer decoder-only Transformer with 12 attention heads and 768-dimensional states, trained with a context window of 512 tokens.”

(Section 3, Model Architecture.)

So:

𝑑
𝑚
𝑜
𝑑
𝑒
𝑙
=
768
d
model
	​

=768

𝑛
ℎ
𝑒
𝑎
𝑑
𝑠
=
12
n
heads
	​

=12

⇒ 
𝑑
ℎ
𝑒
𝑎
𝑑
=
768
/
12
=
64
d
head
	​

=768/12=64

✅ Thus GPT-1 also uses 64-dimensional heads, exactly like Vaswani et al.

⚡ 3. From GPT-2 (2019)

In the GPT-2 paper (Language Models are Unsupervised Multitask Learners), they include this table:

Parameters	Layers	d_model
117M	12	768
345M	24	1024
762M	36	1280
1542M	48	1600


Dimension of head output
-------------------------------------------------------------------
They don’t explicitly list the number of heads or per-head size, but the architecture is said to be “the same as GPT” (i.e., GPT-1).

Since GPT-1 had d_head = 64, we can infer the same pattern holds — and indeed, this is confirmed 
by the released GPT-2 code.


Positions of the dropout layers:
---------------------------------------------------------------------

The original Transformer paper (Vaswani et al., 2017)

In Section 5.4 “Regularization”, Vaswani et al. specify:

“We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.”
“We also apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.”

So for each Transformer sublayer (whether it’s Multi-Head Attention or the Feed-Forward Network):
x = x + Dropout(sublayer(x))

🔍 Dropout placements in the original Transformer

After self-attention output (before residual addition)

After feed-forward output (before residual addition)

After adding positional embeddings to token embeddings

(Optionally) on attention weights (dropout on attention probabilities)


🚀 2. GPT-1 (2018) — “Improving Language Understanding by Generative Pre-Training”

The GPT-1 paper says in Appendix A (“Model Specifications”):

“We use the same hyperparameter settings and initialization scheme as the original Transformer decoder. We modify it to use only the decoder blocks, add layer normalization, and replace ReLU with GELU activations.”

and explicitly states:

“Dropout was applied after every layer and on the attention weights.”

So GPT kept all dropout placements from Vaswani et al. and added one on attention probabilities.

✅ GPT-1 dropout locations

We can infer the following dropouts:

After token + positional embeddings

On attention weights (in the softmaxed attention matrix)

After attention output projection

After feed-forward output projection

and possibly dropout rate = 0.1, same as the original Transformer (unless tuned).
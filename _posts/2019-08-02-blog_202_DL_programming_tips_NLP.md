---
layout: post
title:  "Practical Deep Learning (NLP)"
date:   2019-08-02 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

# How to Classify Text using PyTorch and TorchText?

1.Why PyTorch for Text Classification?

:atom_symbol: **Dealing with Out of Vocabulary words:** A text classification model is trained on fixed vocabulary size. But during inference, we might come across some words which are not present in the vocabulary. These words are known as `Out of Vocabulary` words. Skipping Out of Vocabulary words can be a critical issue as this results in the loss of information.

In order to handle the Out Of Vocabulary words, PyTorch supports a cool feature that replaces the rare words in our training data with Unknown token `UNK`. This, in turn, helps us in tackling the problem of Out of Vocabulary words.

:atom_symbol: **Handling Variable Length sequences:**  PyTorch comes with a useful feature  ‘Packed Padding sequence‘ that implements Dynamic Recurrent Neural Network.

Padding is a process of adding an extra token called padding token at the beginning or end of the sentence. As the number of the words in each sentence varies, we convert the variable length input sentences into sentences with the same length by adding padding tokens. As you can see in the diagram (below), the last element, which is a padding token is also used while generating the output. This is taken care of by the Packed Padding sequence in PyTorch.

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/01/Untitled-Diagram.png)

Packed padding ignores the input timesteps with padding token. These values are never shown to the Recurrent Neural Network which helps us in building a dynamic Recurrent Neural Network.

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/01/Untitled-Diagram1.png)

:atom_symbol: **Wrappers and Pre-trained models:** The state of the art architectures are being launched for PyTorch framework. Hugging Face released Transformers which provides more than 32 state of the art architectures for the Natural Language Understanding Generation!.

## How to preprocess the text data

There are $2$ different types of field objects – `Field` and `LabelField`. Let us quickly understand the difference between the two-


- `Field`: Field object from data module is used to specify preprocessing steps for each column in the dataset.
- `LabelField`: LabelField object is a special case of Field object which is used only for the classification tasks. Its only use is to set the `unk_token` and `sequential` to `None` by default.

**Parameters of Field:**

- Tokenize: specifies the way of tokenizing the sentence i.e. converting sentence to words. I am using spacy tokenizer since it uses novel tokenization algorithm
- Lower: converts text to lowercase
- batch_first: The first dimension of input and output is always batch size


```py
# TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
# LABEL = data.LabelField(dtype = torch.float,batch_first=True)
```

TODO: Implement it using vanila PyTorch and then refactor using `PyTorchLightning`

**Reference:**

- [Build Your First Text Classification model using PyTorch](https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/) :fire:


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# How FAISS index embedding and do search on them?

`Faiss` is a library for efficient `similarity search` and `clustering of dense vectors`. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy.

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

**Efficiency:** Most of the methods, like those based on `binary vectors` and **compact quantization codes**, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server.

The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace IndexFlatL2 with GpuIndexFlatL2) and copies to/from GPU memory are handled automatically. Results will be faster however if both input and output remain resident on the GPU. Both single and multi-GPU usage is supported.

## How FAISS works?

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to


- Search time
- Search quality
- Memory used per index vector
- Training time
- Need for external data for unsupervised training

The optional GPU implementation provides what is likely (as of March 2017) the fastest exact and approximate (compressed-domain) nearest neighbor search implementation for high-dimensional vectors, fastest `Lloyd's k-means`, and fastest small k-selection algorithm known.

## Under the hood

The Facebook AI Research team started developing Faiss in 2015, based on research results and a substantial engineering effort. For this library, we chose to focus on properly optimized versions of a few fundamental techniques. In particular, on the CPU side we make heavy use of:

- **Multi-threading** to exploit multiple cores and perform parallel searches on multiple GPUs.
- **BLAS** libraries for efficient exact distance computations via matrix/matrix multiplication. An efficient brute-force implementation cannot be optimal without using BLAS. BLAS/LAPACK is the only mandatory software dependency of Faiss.
- Machine **SIMD** vectorization and popcount are used to speed up distance computations for isolated vectors.

**Reference:**

- [Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) :fire: :fire:


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


# Difference of `Conv1D` and `Conv2D` in deep learning. or How CNN works on text data?

**Conv2D**

In simple terms, images have shape `(height, width)`. So a filter can move in 2 direction, so `conv2D` is used.

![image](https://miro.medium.com/max/700/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)

In the above image, the green matrix is the `kernel` convolving on `2 direction` over the image and creating the red `feature map` in the right

**Conv1D**
But in case of text, initially the text are converted to some fixed dimension vectors `one-hot-encoded` or `dense embedding` of fixed dimension. Where the filters can move in one direction only, i.e in the direction of the words or characters, but not in the corresponding embedding dimension because it's fixed.   

![image](https://debajyotidatta.github.io/assets/images/conv.001.png)

- So the green boxes represent the words or the characters depending on your approach. 
- And the corresponding blue rows shows the vector representation (one-hot-encoding or embedding) of the words or the characters.

Here is a corresponding kernel whose `height=kernel size` but it's `widht=embedding_dim` which is fixed. 

![kernel](https://debajyotidatta.github.io/assets/images/conv.002.png)

So the above kernel can move along the direction of the words or characters, i.e, along the green boxes in the previous image.

> **Convolve is a fancy term for multiplication with corresponding cells and adding up the sum.** 

It varies based on things like 1. `stride` (How much the filter moves every stage?) and the 2. `length` of the filter. The output of the convolution operation is directly dependent on these two aspects. 

_first convolution_

![image](https://debajyotidatta.github.io/assets/images/conv.003.png)

_last convolution_

![image](https://debajyotidatta.github.io/assets/images/conv.006.png)

See after each `stride` a single cell is generated at the right and after the full pass, a 1D vector is generated.

Now if multiple convolution filters are used, then multiple such `1D vectors` will be generated. Then you do `maxpooling` to get the `max element` from each such `1D` vector and then soncatenate and finally apply `softmax`.

_multiple feature maps due to multiple kernels_

![image](https://debajyotidatta.github.io/assets/images/conv2.006.png)

_max pooling and concatenation_

![image](https://debajyotidatta.github.io/assets/images/conv2.007.png)

The entire process was very nicely illustrated by Zhang et al, in the paper “A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification”, for words.


![image](https://debajyotidatta.github.io/assets/images/Zhang.png)


**Reference:**

- [Understanding Convolutions in Text](https://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/11/27/Understanding-Convolutions-In-Text/)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Text Classification in Pytorch

PyTorchNLPBook by Delip Rao, Chapter 3

- [Classifying_Yelp_Review_Sentiment](https://nbviewer.jupyter.org/github/msank00/nlproc/blob/master/Classifying_Yelp_Reviews.ipynb)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----


# Programming Tips for RNN, LSTM, GRU in Pytorch


## LSTM


![image](/assets/images/image_24_lstm_1.png)

**Problem definition:** Given family name, identify nationality

```py
class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input_, hidden):
        """
        input_.view(ni, nb, -1): 
            ni: Number of inputs given simultaneously
                (this helps in vectorization)
            nb: number of batches
            -1: rest, here number of characters 
        """
        
        out, hidden = self.lstm_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output.view(1, -1), hidden
    
    def init_hidden(self):
        """
        Return:
            Tuple of size 2
            - initializing hidden state
            - initializing cell state
        """
        # dim: n_layers x n_batches x hid_dim
        init_hidden_state = torch.zeros(1, 1, self.hidden_size)

        # dim: n_layers x n_batches x hid_dim
        init_cell_state = torch.zeros(1, 1, self.hidden_size) 
        return (init_hidden_state, init_cell_state)

n_hidden = 128
net = LSTM_net(n_letters, n_hidden, n_languages)
```

- one parameter update per batch

**code source**: Deep Learning course from PadhAI, IIT M, Module: Sequence Model, Lecture: Sequence model in Pytorch

From PyTorch `nn.LSTM` [documentation](https://pytorch.org/docs/stable/nn.html)

```py
torch.nn.LSTM(*args, **kwargs)
```

Applies a **multi-layer** `long short-term memory` (LSTM) RNN to an input sequence.



- $i_t​=\sigma(W_{ii}​x_t​+b_{ii}​+W_{hi}​h_{(t−1)}​+b_{hi}​)$
- $f_t​=\sigma(W_{if}​x_t​+b_{if}​+W_{hf}​h_{(t−1)}​+b_{hf}​)$
- $g_t​=\tanh(W_{ig}​x_t​+b_{ig}​+W_{hg}​h_{(t−1)}​+b_{hg}​)$
- $o_t​=\sigma(W_{io}​x_t​+b_{io}​+W_{ho}​h_{(t−1)}​+b_{ho}​)$
- $c_t​=f_t​∗c_{(t−1)​}+i_t​∗g_t$
- $h_t = o_t*\tanh(c_t)$

**Parameters:**

- `input_size` – The number of expected features in the input `x`
- `hidden_size` – The number of `features/neurones` in the hidden state `h`
- `num_layers` – Number of recurrent layers. E.g., setting `num_layers=2` would mean **stacking two LSTMs** together to form a `stacked LSTM`, with the second LSTM taking in outputs of the first LSTM and computing the final results. `Default: 1`


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# What's the difference between `hidden` and `output` in PyTorch LSTM?

According to Pytorch documentation 

```py
"""
Outputs: output, (h_n, c_n)
"""
```

- `output (seq_len, batch, hidden_size * num_directions)`: Tensor containing the output features (h_t) from the last layer of the RNN, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
- `h_n (num_layers * num_directions, batch, hidden_size)`: tensor containing the hidden state for t=seq_len
- `c_n (num_layers * num_directions, batch, hidden_size)`: tensor containing the cell state for t=seq_len


## How to interpret it?

output comprises all the hidden states in the last layer ("last" depth-wise, not time-wise). $(h_n, c_n)$ comprises the hidden states after the last time step, $t = n$, so you could potentially feed them into another LSTM.

![image](https://i.stack.imgur.com/SjnTl.png)

The batch dimension is not included.

- [source_stackOverflow](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm)


**Remember**

For each element in the input sequence, each layer computes the following function:



- The `RNN_Net` and the `LSTM_Net` should be equivalent from outside, i.e their `function signature` should be equivalent meaning their input and output signature are equivalent even if their internal mechanism is different

```py
"""
input_dim:int = size of the input vectors depending on problem definition. It can be number of words or number of characters etc.
hid_dim:int = size of the hidden dimension, i.e number of neurons in the hidden layer, you SHOULD NOT interpret this as number of hidden layer
output_dim:int = size of the output, it's mostly size of the multi-class vector, e.g: number of language, number of sentiments etc.
"""
net_rnn = RNN_net(input_dim, hidden_dim, output_dim)
net_lstm = RNN_net(input_dim, hidden_dim, output_dim)
```

- Both should return `output` and `hidden` state


**Reference:**

- Deep Learning course from PadhAI, IIT M, Module: Sequence Model, Lecture: Sequence model in Pytorch


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


# Pytorch RNN tips, set `batch_first=True`:

Always set `batch_first=True` while implementing RNN using PyTorch RNN module. 

**Reference:**

- [Beginners guide on RNN](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Variable sequence lengths in RNN using Pytorch

Minimal tutorial on packing (pack_padded_sequence) and unpacking (pad_packed_sequence) sequences in pytorch.


- [Important Notes](https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec)
- [More readable blog](https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial)

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
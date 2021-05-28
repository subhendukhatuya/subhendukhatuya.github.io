---
layout: post
title:  "Deep Learning Papers"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

List of papers - you must read

# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

**Summary:** Deep neural nets with a large number of parameters are very powerful machine learning
systems. However, overfitting is a serious problem in such networks. Large networks are also
slow to use, making it difficult to deal with overfitting by combining the predictions of many
different large neural nets at test time. Dropout is a technique for addressing this problem.
The key idea is to randomly drop units (along with their connections) from the neural
network during training. This prevents units from co-adapting too much. During training,
dropout samples from an exponential number of different “thinned” networks. At test time,
it is easy to approximate the effect of averaging the predictions of all these thinned networks
by simply using a single unthinned network that has smaller weights. This significantly
reduces overfitting and gives major improvements over other regularization methods. We
show that dropout improves the performance of neural networks on supervised learning
tasks in vision, speech recognition, document classification and computational biology,
obtaining state-of-the-art results on many benchmark data sets.

JMLR 2014

[paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Summary:** Training Deep Neural Networks is complicated
by the fact that the distribution of each layer’s
inputs changes during training, as the parameters
of the previous layers change. This slows
down the training by requiring lower learning
rates and careful parameter initialization, and
makes it notoriously hard to train models with
saturating nonlinearities. We refer to this phenomenon
as internal covariate shift, and address
the problem by normalizing layer inputs.
Our method draws its strength from making normalization
a part of the model architecture and
performing the normalization for each training
mini-batch. Batch Normalization allows us to
use much higher learning rates and be less careful
about initialization, and in some cases eliminates
the need for Dropout. Applied to a stateof-the-art
image classification model, Batch Normalization
achieves the same accuracy with 14
times fewer training steps, and beats the original
model by a significant margin. Using an ensemble
of batch-normalized networks, we improve
upon the best published result on ImageNet classification:
reaching 4.82% top-5 test error, exceeding
the accuracy of human raters.

ICML 2015

[paper](http://proceedings.mlr.press/v37/ioffe15.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Generative Adversarial Nets

**Summary:** We propose a new framework for estimating generative models via an adversarial
process, in which we simultaneously train two models: a generative model G
that captures the data distribution, and a discriminative model D that estimates
the probability that a sample came from the training data rather than G. The training
procedure for G is to maximize the probability of D making a mistake. This
framework corresponds to a minimax two-player game. In the space of arbitrary
functions G and D, a unique solution exists, with G recovering the training data
distribution and D equal to `0.5` everywhere. In the case where G and D are defined
by multilayer perceptrons, the entire system can be trained with backpropagation.
There is no need for any Markov chains or unrolled approximate inference networks
during either training or generation of samples. Experiments demonstrate
the potential of the framework through qualitative and quantitative evaluation of
the generated samples.

NIPS 2014 

[paper](http://datascienceassn.org/sites/default/files/Generative%20Adversarial%20Nets.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Attention Is All You Need

**Summary:**The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German
translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.0 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature.

NIPS 2017
[paper](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

----

# How transferable are features in deep neural networks?

**Summary:**Many deep neural networks trained on natural images exhibit a curious phenomenon
in common: on the first layer they learn features similar to Gabor filters
and color blobs. Such first-layer features appear not to be specific to a particular
dataset or task, but general in that they are applicable to many datasets and tasks.
Features must eventually transition from general to specific by the last layer of
the network, but this transition has not been studied extensively. In this paper we
experimentally quantify the generality versus specificity of neurons in each layer
of a deep convolutional neural network and report a few surprising results. Transferability
is negatively affected by two distinct issues: (1) the specialization of
higher layer neurons to their original task at the expense of performance on the
target task, which was expected, and (2) optimization difficulties related to splitting
networks between co-adapted neurons, which was not expected. In an example
network trained on ImageNet, we demonstrate that either of these two issues
may dominate, depending on whether features are transferred from the bottom,
middle, or top of the network. We also document that the transferability of features
decreases as the distance between the base task and target task increases, but
that transferring features even from distant tasks can be better than using random
features. A final surprising result is that initializing a network with transferred
features from almost any number of layers can produce a boost to generalization
that lingers even after fine-tuning to the target dataset.


NIPS 2014
[paper](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# k-means++: The Advantages of Careful Seeding

**Summary**: The k-means method is a widely used clustering technique that seeks to minimize the average
squared distance between points in the same cluster. Although it offers no accuracy guarantees,
its simplicity and speed are very appealing in practice. By augmenting k-means with a simple,
randomized seeding technique, we obtain an algorithm that is O(log k)-competitive with the
optimal clustering. Experiments show our augmentation improves both the speed and the
accuracy of k-means, often quite dramatically.

[paper](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)



# References

1. [Awsome deep learning papers](https://github.com/terryum/awesome-deep-learning-papers#understanding--generalization--transfer)
2. [Papers from NIPS 2016](https://github.com/solaris33/awesome-machine-learning-papers)
3. [Detailed list of papers, datasets, etc.](https://github.com/ChristosChristofidis/awesome-deep-learning)
4. [Concise List](http://www.xn--vjq503akpco3w.top/literature/awesome-free-deep-learning-papers.html)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>


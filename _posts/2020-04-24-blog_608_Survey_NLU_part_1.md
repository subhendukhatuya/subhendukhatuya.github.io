---
layout: post
title:  "Survey - Natural Language Understanding (NLU - Part 1)"
date:   2020-04-24 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

Quick Refresher on Natural Language Understanding

# What is Syntactic and Semantic analysis?

Syntactic analysis (syntax) and semantic analysis (semantic) are the two primary techniques that lead to the understanding of natural language. Language is a set of valid sentences, but what makes a sentence valid? Syntax and semantics.

- **Syntax** is the grammatical structure of the text 
- **Semantics** is the meaning being conveyed. 

A sentence that is syntactically correct, however, is not always semantically correct. 
- **Example,** “cows flow supremely” is grammatically valid (subject — verb — adverb) but it doesn't make any sense.

## SYNTACTIC ANALYSIS

Syntactic analysis, also referred to as syntax analysis or parsing. 

> It is the process of analyzing natural language with the rules of a formal grammar. 

Grammatical rules are applied to categories and groups of words, not individual words. Syntactic analysis basically assigns a semantic structure to text.

For example, a sentence includes a subject and a predicate where the subject is a noun phrase and the predicate is a verb phrase. Take a look at the following sentence: “The dog (noun phrase) went away (verb phrase).” Note how we can combine every noun phrase with a verb phrase. Again, it's important to reiterate that a sentence can be syntactically correct but not make sense.

## SEMANTIC ANALYSIS

The way we understand what someone has said is an unconscious process relying on our intuition and knowledge about language itself. In other words, the way we understand language is heavily based on meaning and context. Computers need a different approach, however. The word `semantic` is a **linguistic term** and means `related to meaning or logic`.

:paperclip: **Reference:**

- [Blog](https://builtin.com/data-science/introduction-nlp)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# What is Natural Language Underrstanding?

It can be easily understood by the syllabus topic of the course CS224U by Standford. Though over the years the definition has been changed.

**2012**

- WordNet
- Word sense disambiguation
- Vector-space models
- Dependency parsing for NLU
- Relation extraction
- Semantic role labeling
- Semantic parsing
- Textual inference
- Sentiment analysis
- Semantic composition withvectors
- Text segmentation
- Dialogue

**2020**

- Vector-space models
- Sentiment analysis
- Relation extraction
- Natural LanguageInference
- Grounding
- Contextual wordrepresentations
- Adversarial testing
- Methods and metrics

:paperclip: **Reference:**

- [CS224u course website](https://web.stanford.edu/class/cs224u/)
- [CS224u slide](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-intro-handout.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Motivation: Why Learn Word Embeddings?

Image and audio processing systems work with rich, high-dimensional datasets encoded as vectors of the individual raw pixel-intensities for image data, or e.g. power spectral density coefficients for audio data. For tasks like object or speech recognition we know that all the information required to successfully perform the task is encoded in the data (because humans can perform these tasks from the raw data). However, **natural language processing** systems **traditionally treat words as discrete atomic symbols**, and therefore `cat` may be represented as `Id537` and `dog` as `Id143`. These encodings are **arbitrary, and provide no useful information** to the system regarding the relationships that may exist between the individual symbols. This means that the model can leverage very little of what it has learned about ‘cats’ when it is processing data about ‘dogs’ (such that they are both animals, four-legged, pets, etc.). Representing words as unique, discrete ids furthermore leads to **data sparsity**, and usually means that we may need more data in order to successfully train statistical models. Using vector representations can overcome some of these obstacles.

**Vector space models** (**VSM**s) represent (`embed`) words in a continuous vector space where **semantically similar** (meaningfully similar) words are mapped to nearby points (`are embedded nearby each other`). VSMs have a long, rich history in NLP, but all methods depend in some way or another on the **Distributional Hypothesis**, 

> which states that words that appear in the same contexts share semantic meaning. 

The different approaches that leverage this principle can be divided into two categories: 

- **Count-based methods** (e.g. Latent Semantic Analysis),
- **Predictive methods** (e.g. neural probabilistic language models like `word2vec`).

This distinction is elaborated in much more detail by [Baroni et al. 2014](https://www.aclweb.org/anthology/P14-1023.pdf) in his great paper **Don’t count, predict!**, where he compares the `context-counting` vs `context-prediction`. In a nutshell: 
  - **Count-based** methods first compute the statistics of how often some word co-occurs with its neighbor words in a large text corpus, and then map these count-statistics down to a small, dense vector for each word. 
  - **Predictive models** `directly try to predict` a word from its neighbors in terms of learned small, dense embedding vectors (considered parameters of the model).

**Word2vec** is a particularly **computationally-efficient predictive model** for learning word embeddings from raw text. It comes in two flavors, the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. 
  - Algorithmically, these models are similar, except that CBOW predicts target words (e.g. ‘mat’) from source context words (‘the cat sits on the’), while the skip-gram does the inverse and predicts source context-words from the target words. 

This inversion might seem like an arbitrary choice, but statistically it has different effect.
- **CBOW** **smoothes over a lot of the distributional information** (by treating an entire context as one observation). For the most part, this turns out to be a **useful thing for smaller datasets**. 
- **Skip-gram** treats each context-target pair as a new observation, and this tends to **do better when we have larger datasets**. 

:paperclip: **Reference:**

- [Tensorflow: Vector Representations of Words](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/tutorials/word2vec/index.md) :fire:
- [Baroni et al. 2014](https://www.aclweb.org/anthology/P14-1023.pdf)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to design a basic vector space model?

- [Youtube](https://www.youtube.com/watch?v=gtuhPq0Xyno&feature=youtu.be)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# What is PMI: Point-wise Mutual Information?

The idea behind the NLP algorithm is that of transposing words into a vector space, where each word is a D-dimensional vector of features. By doing so, we can compute some quantitative metrics of words and between words, namely their cosine similarity.

**Problem:** How to understand whether two (or more) words actually form a unique concept?

**Example:** Namely, consider the expression ‘social media’: both the words can have independent meaning, however, when they are together, they express a precise, unique concept.

Nevertheless, it is not an easy task, since if both words are frequent by themselves, their co-occurrence might be just a chance. Namely, consider the name ‘Las Vegas’: it is not that frequent to read only ‘Las’ or ‘Vegas’ (in English corpora of course). The only way we see them is in the bigram Las Vegas, hence it is likely for them to form a unique concept. On the other hand, if we think of ‘New York’, it is easy to see that the word ‘New’ will probably occur very frequently in different contexts. How can we assess that the co-occurrence with York is meaningful and not as vague as ‘new dog, new cat…’?

The answer lies in the **Pointwise Mutual Information (PMI)** criterion. The idea of PMI is that we want to 

> quantify the likelihood of co-occurrence of two words, taking into account the fact that it might be caused by the frequency of the single words. 

Hence, the algorithm computes the ($\log$) probability of co-occurrence scaled by the product of the single probability of occurrence as follows:

<center>

$
PMI(w_a, w_b) = \log  \left( \frac{p(w_a, w_b)}{p(w_a) p(w_b)} \right) = \log \left( \frac{p(w_a, w_b)}{p(w_a)} \times \frac{1}{p(w_b)} \right)
$

</center>

<center>

$
= \log \left( p(w_b \vert w_a) \times \frac{1}{p(w_b)} \right) = \log \left( p(w_a \vert w_b) \times \frac{1}{p(w_a)} \right) 
$

</center>

where $w_a$ and $w_b$ are two words.

Now, knowing that, when $w_a$ and $w_b$ are independent, their joint probability is equal to the product of their marginal probabilities, when the ratio equals 1 (hence the log equals 0), it means that the two words together don’t form a unique concept: they co-occur by chance.

On the other hand, if either one of the words (or even both of them) has a **low probability of occurrence if singularly considered**, but **its joint probability together with the other word is high**, it means that the two are likely to express a **unique concept**.

> PMI is the re-weighting of the entire count matrix

Let’s focus on the last expression. As you can see, it’s the conditional probability of $w_b$ given $w_a$ times $\frac{1}{p(w_b)}$. If $w_b$ and $w_a$ are independent, there is no meaning to the multiplication (it’s going to be zero times something). But if the conditional probability is larger than zero, $p(w_b \vert w_a) > 0$, then there is a meaning to the multiplication. How `important` is the event $W_b = w_b$? if $P(W_b = w_b) = 1$ then the event $W_b = w_b$ is not really important is it? think a die which always rolls the same number; there is no point to consider it. But, If the event $W_b = w_b$ is fairly rare → $p(w_b)$ is relatively low → $\frac{1}{p(w_b)}$ is relatively high → the value of $p(w_b \vert w_a)$ becomes much more important in terms of information. So that is the first observation regarding the PMI formula. 

:paperclip: **Reference:**

- [PMI](https://medium.com/dataseries/understanding-pointwise-mutual-information-in-nlp-e4ef75ecb57a)
- [understanding-pointwise-mutual-information-in-statistics](https://eranraviv.com/understanding-pointwise-mutual-information-in-statistics/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# LSA: Latent Semantic Analysis

**Note:** LSA and [LS (Latent Semantic Indexing) are mostly used synonymously, with the information retrieval community usually referring to it as LSI. LSA/LSI uses SVD to decompose the term-document matrix $A$ into a term-concept matrix $U$, a singular value matrix $S$, and a concept-document matrix $V$ in the form: $A = USV'$

- One of the oldest and most widely used dimensionality reduction method.
- Also known as **truncated SVD**
- Standard baseline, often very tough to beat.

## Related dimensionality reduction technique

- PCA
- NNMF
- Probabilistic LSA
- LDA
- t-SNE

For more details check this [blog from msank](https://msank00.github.io/blog/2019/08/05/blog_203_ML_NLP_Part_1#Top)


:paperclip: **Reference:**

- [Blog: LSI](https://msank00.github.io/blog/2019/08/05/blog_203_ML_NLP_Part_1#what-is-lsi-latent-semantic-indexing)
- [CS224U Youtube Lecture](https://www.youtube.com/watch?v=nH4rn3X8i0c&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=3) :rocket:

-----

# GloVe: Global Vectors

Read this amazing paper by **Pennington et al.** (2014) [[1]](#1)

GloVe [[1]](#1) is an **unsupervised learning algorithm** for obtaining vector representations for words. Training is performed on aggregated global `word-word` **co-occurrence statistics** from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

>> The authors of Glove show that the ratio of the co-occurrence probabilities of two words (rather than their co-occurrence probabilities themselves) is what contains information and aim to encode this information as vector differences.

To achieve this, they propose a **weighted least squares objective** $J$ that directly aims to minimize the difference between the dot product of the vectors of two words and the logarithm of their number of co-occurrences:

<center>

$
J = \sum\limits_{i, j=1}^V f(X_{ij})   (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \text{log}   X_{ij})^2
$

</center>


where $w_i$ and $b_i$ are the word vector and bias respectively of word $i$, $\tilde{w_j}$ and $b_j$ are the context word vector and bias respectively of word $j$, $X_{ij}$ is the number of times word $i$ occurs in the context of word $j$, and $f$ is a weighting function that assigns relatively lower weight to rare and frequent co-occurrences.

**Main Idea:**

> The objective is to learn vectors for words such that their **dot product is proportional to their probability of co-occurrence**.

- Can use the `Mittens` package. [PyPi](https://pypi.org/project/mittens/), [Paper](https://www.aclweb.org/anthology/N18-2034/)

<center>
<img src="/assets/images/image_40_nlu_01.png" alt="image" width="500">
</center>

- $w_i$: row embedding
- $w_k$: column embedding
- $X_{ik}$: Co-occurrence count
- $\log(P_{ik})$: log of co-occurrence probability
- $\log(X_{ik})$: log of co-occurrence count
- $\log(X_i)$: log of row probability

Their dot products are the 2 primary terms + 2 bias terms.

And the idea is that should be equal to (at-least proportional to) the log of the co-occurrence probability. 

Equation 6 tells that the dot product is equal to the difference of  of two log terms and if re-arrange them they looks very similar to **PMI** !! Where PMI is the re-weighting of the entire count matrix. 

## The Weighted GloVe objective


<center>
<img src="/assets/images/image_40_nlu_02.png" alt="image" height="250">
</center>


<center>
<img src="/assets/images/image_40_nlu_03.png" alt="image" width="300">
</center>

Weighted by the function $f()$. Which is `flatten`ing out and `rescale`ing the co-occurrence count $X_{ik}$ values.

Say the co-occurrence count vector is like this `v = [100 99 75 10 1]`. Then $f(v)$ is `[1.00 0.99 0.81 0.18 0.03]`.

## What's happening behind the scene (BTS)?

<center>
<img src="/assets/images/image_40_nlu_04.png" alt="image" width="600">
</center>

**Example:**

Word `wicked` and `gnarly` (positive slang) never co-occur. If you look at the left plot in the above image, then you see, what GLoVe does is, it pushes both `wicked` and `gnarly` **away from negative word** `terrible` and moves them **towards positive word** `awsome`. Because even if `wicked` and `gnarly` don't occur together, they have co-occurrence with positive word `awsome`. GloVe thus achieves this latent connection.   

**Note:** Glove transforms the `raw count` distribution into a `normal distribution` which is essential when you train deep-learning model using word-embedding as your initial layer. It's essential because the embedding values have constant `mean` and `variance` and this is a crucial part for training any deep-learning model. The weight values while passing through different layers should maintain their distribution. That's why GloVe does so well as an input to another system. 

:paperclip: **Reference:**

- [CS224U Slide](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-vsm-handout.pdf) :pushpin:
- [CS224U Youtube](https://www.youtube.com/watch?v=pip8h9vjTHY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=4) :pushpin:
- <a id="1">[1]</a> [Paper: Glove: Global Vectors for Word Representation, EMNLP 2014]((https://www.aclweb.org/anthology/D14-1162.pdf)) :fire:
- [On word embeddings - Part 1: Sebastian Ruder](https://ruder.io/word-embeddings-1/index.html)
- [Language Model: CS124](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Word2Vec

- Introduced by Mikolov et al. 2013 [[2]](#2)
- Goldberg and Levy 2014 [[3]](#3) identified the relation between `word2vec` and `PMI`
- [Gensim](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) package has a highly scalable implementation

We are going to summarize $2$ papers 

- **Efficient Estimation of Word Representations in Vector Space** [[1]](#1)
  - Author: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
  - [Paper](https://arxiv.org/abs/1301.3781)
- **Distributed Representations of Words and Phrases and their Compositionality** [[2]](#2)
  - Author: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean 
  - [Paper](https://arxiv.org/abs/1310.4546)

----


- `Distributed representations of words` in a `vector space` help learning algorithms to achieve better performance in natural language processing tasks by `grouping similar words`. 
- Earlier there were label encoding, binarization or one-hot-encoding. Those were `sparse vector representation`. But the Skip-gram model helps to represent the words as `continuous dense vector`.
- The problem with label encoding, one-hot-encoding type vector representation is that, they don't capture the correlation (very loosely speaking) with each other. The correlation groups the words in terms of their hidden or latent meaning.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Example

Let's say we have 4 words: `dog`, `cat`, `chair` and `table`.

If we apply one-hot-encoding then:

- dog: [0, 0]
- cat: [0, 1]
- chair: [1, 0]
- table: [1, 1]

The above is a random order. We can put in any random order that will not be a problem. Because all 4 words are not related in the vector space.


<center>
<img src="/assets/images/image_33_w2v_1.png" alt="image" height="300">
</center>

However `cat` and `dog` are from type _animal_ and `chair` and `table` are from type _furniture_. So it would have been very good if in the vector space they were grouped together and their vectors are adjusted accordingly.

<center>
<img src="/assets/images/image_33_w2v_2.png" alt="image" height="300">
</center>

Now there are many methods to learn these kind of representation. But at present 2 popular ones are `CBOW: Continuous Bag of Words` and `Skip-gram` model.

Both are kind of complement of each other. 

Say, we have a sentence $S = w_1 w_2 w_3 w_4 w_5$ where $w_i$ are words. Now say we pick word $w_3$ as our `candidate word` for which we are trying to get the dense vector. Then the remaining words are `neighbor words`. These neighbor words denote the context for the candidate words and the dense vector representation capture these properties.  

- Sentence $S = w_1 w_2 w_3 w_4 w_5$
- Candidate: $w_3$
- Neighbor:  $w_1 w_2 w_4 w_5$

## Objective 

The train objective is to learn word vector representation that are good at predicting the nearby words.

**CBOW Objective:** Predicts the candidate word $w_3$ based on neighbor words $w_1 w_2 w_4 w_5$. 


**Skip-gram Objective:** Predicts the Neighbor words $w_1 w_2 w_4 w_5$ based on candidate word $w_3$


![image](/assets/images/word2vec_3.png)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Word2Vec

Let us now introduce arguably the most popular word embedding model, the model that launched a thousand word embedding papers: **word2vec**, the subject of two papers by Mikolov et al. in 2013. 

As word embeddings are a key building block of deep learning models for NLP, word2vec is often assumed to belong to the same group. Technically however, word2vec is not be considered to be part of deep learning, as its architecture is neither deep nor uses non-linearities (in contrast to Bengio's model and the C&W model).

In their first paper [[4]](#4), Mikolov et al. propose **two architectures** for learning word embeddings that are computationally less expensive than previous models. 

In their second paper [[2]](#2), they improve upon these models by employing additional strategies to enhance training speed and accuracy.

These architectures offer two main benefits over the C&W model [[6]](#6) and Bengio's language model [[5]](#5):

- They do away with the expensive hidden layer.
- They enable the language model to take additional context into account.

As we will later show, the success of their model is not only due to these changes, but especially due to certain training strategies.

**Side-note:** `word2vec` and `GloVe` might be said to be to NLP what VGGNet is to vision, i.e. a common weight initialization that provides generally helpful features without the need for lengthy training.



In the following, we will look at both of these architectures:

## Continuous bag-of-words (CBOW)

Mikolov et al. thus use both the $n$ words before and after the target word $w_t$ to predict it. They call this continuous bag-of-words (CBOW), as it uses continuous representations whose order is of no importance.

The objective function of CBOW in turn is only slightly different than the language model one:

<center>

$
J_\theta = \frac{1}{T}\sum\limits_{t=1}^T\ \log p(w_t \vert w_{t-n}^{t-1},w_{t+1}^{t+n})
$

</center>

where $w_{t-n}^{t-1}=w_{t-n} , \cdots , w_{t-1}$ and $w_{t+1}^{t+n}=w_{t+1}, \cdots , w_{t+n}$ 

## The Skip-gram model

While CBOW can be seen as a precognitive language model, skip-gram turns the language model objective on its head: Instead of using the surrounding words to predict the centre word as with CBOW, skip-gram uses the centre word to predict the surrounding words

The training objective of the Skip-gram model is to find word representations that are useful for predicting the surrounding words in a sentence or a document. More formally, given a sequence of training words $w_1, w_2, w_3, \dots , w_T$, the objective of the Skip-gram model is to maximize the average log probability, i.e

<center>

$J_\theta = \frac{1}{T} \sum\limits_{t=1}^T \sum\limits_{-c \leq j \leq c , j \ne 0} \log{p(w_{t+j}\vert w_t)}$

</center>

where $c$ is the size of the training context (which can be a function of the center word $w_t$). Larger $c$ results in more training examples and thus can lead to a higher accuracy, at the expense of the training time.

The basic Skip-gram formulation defines $p(w_{t+j} \vert w_t)$ using the softmax function. 

- A computationally `efficient approximation` of the full softmax is the **hierarchical softmax**. The main advantage is that instead of evaluating $W$ output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about $\log_2(W)$ nodes.
- The hierarchical softmax uses a binary tree representation of the output layer with the W words as its leaves and, for each node, explicitly represents the relative probabilities of its child nodes. These define a random walk that assigns probabilities to words.
- Such representation makes the learning faster using distributed technique.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Word2vec: from corpus to labeled data

- input sentence: `it was the best of times it was the worst of times...`
- window size = 2

<center>
<img src="/assets/images/image_40_nlu_07.png" alt="image" height="200">
</center>

Earlier in LSA and others, it was **count-based co-occurrence**, but here it's like positional co-occurrence, thus bypassing the task of creating the count-matrix.

<center>
<img src="/assets/images/image_40_nlu_05.png" alt="image" width="500">
</center>

- $C$ is a label vector for individual example and it's one-hot-encoded. But it has the dimensionality of the entire Vocabulary $V$. Where size of $V$ is very big and therefore, it's difficult to train. But that's the intuition, that after creating a labelled dataset apply standard machine learning classifier to predict those labels. However, this is the backbone of the word2vec variant **SkipGram** model.  

> The objective of the model is to push the `dot product` in a particular direction i.e towards words which co-occur a lot.  

To bypass the training issue a variant of SkipGram is used.

## Word2vec: noise contrastive estimation

<center>
<img src="assets/images/image_40_nlu_06.png" alt="image" width="500">
</center>

Sum of 2 separate objective, each one of them is binary. The left side is for those words which actually co-occur. And then we sample some negative instances (meaning the word pair that doesn't co-occur) which are used in the right side objective. 

:paperclip: **Reference:**

- <a id="2">[2]</a> [Paper: Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- <a id="3">[3]</a> [Paper: Neural Word Embeddingas Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
- <a id="4">[4]</a> 
Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. (ICLR 2013)
- <a id="5">[5]</a> 
Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). A Neural Probabilistic Language Model. The Journal of Machine Learning Research
- <a id="6">[6]</a> 
Collobert, R., & Weston, J. (2008). A unified architecture for natural language processing. ICML ’08,
- <a id="7">[7]</a> 
Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global Vectors for Word Representation, EMNLP 2014
- [CS224u Youtube Lecture](https://www.youtube.com/watch?v=pip8h9vjTHY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=4)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Scaling up with Noise-Contrastive Training

Neural probabilistic language models are traditionally trained using the maximum likelihood (ML) principle to maximize the probability of the next word $w_t$ (for `target`) given the previous words $h$ (for `history`) in terms of a softmax function,

<center>

$
P(w_t \vert h) = \text{softmax}(\text{score}(w_t, h)) = \frac{\exp ({ \text{score}(w_t, h) }) } {\sum_\text{Word w' in Vocab} \exp ({ \text{score}(w', h) }) }
$

</center>

where $\text{score}(w_t, h)$ computes the compatibility of word $w_t$ with the context $h$ (a dot product is commonly used). We train this model by maximizing its log-likelihood on the training set, i.e. by maximizing

<center>

$J_\text{ML} = \log P(w_t \vert h) = \text{score}(w_t, h) - \log \left( \sum_{w' \in V} \exp ({ \text{score}(w', h) }) \right)$

</center>

- $w'$ is a word
- $V$ is the vocabulary set

This yields a properly normalized probabilistic model for language modeling. However this is very **expensive**, because we need to compute and normalize each probability using the score for all other $V$ words $w'$ in the current context $h$, at every training step.

On the other hand, for feature learning in `word2vec` we do not need a full probabilistic model. The CBOW and skip-gram models are **instead trained using a binary classification objective** (logistic regression) to discriminate the real target words $w_t$ from $k$ imaginary (`noise`) words $\tilde w$, in the same context. We illustrate this below for a CBOW model. For skip-gram the direction is simply inverted.

Mathematically, the objective (for each example) is to maximize

<center>

$J_\text{NEG} = \log Q_\theta(D=1 \vert w_t, h) + k \mathop{\mathbb{E}}{\tilde w \sim P_\text{noise}} \left[ \log Q_\theta(D = 0 \vert \tilde w, h) \right]$

</center>

- where $Q_\theta(D=1 \vert w, h)$ is the binary logistic regression probability under the model of seeing the word $w$ in the context $h$ in the dataset $D$, calculated in terms of the learned embedding vectors $\theta$. 
- In practice the author approximates the expectation by drawing $k$ contrastive words (contrasting/different words) from the noise distribution (i.e. we compute a [Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)).

This objective is maximized when the model assigns high probabilities to the real words, and low probabilities to noise words. Technically, this is called `Negative Sampling` [[8]](#8), and there is good mathematical motivation for using this loss function: 
- The updates it proposes **approximate the updates of the softmax function in the limit**. 
- But computationally it is especially appealing because computing the loss function now **scales only with the number of noise words that we select** ($k$), and not all words in the vocabulary ($V$). 
- This makes it much **faster to train**. The author uses the very similar `noise-contrastive estimation` [[9]](#9) (NCE) loss.

:paperclip: **Reference:**

- [TensorFlow: Vector Representations of Words](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/tutorials/word2vec/index.md) :bomb: :fire: :rocket:
- <a id="8">[8]</a> [Paper: Distributed Representations of Words and Phrases
and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- <a id="9">[9]</a> [Paper: Learning word embeddings efficiently with
noise-contrastive estimation](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# Retrofitting

**Central goals:**

- Distributional representations are powerful and easy, but they tend to reflect only similarity (synonymy, connotation).
- Structured resources are sparse and hard to obtain, but they support learning rich, diverse, `semantic` distinctions.
- Can we have the best of both world? Answer is YES using **Retrofitting**.
- Read the original paper from [Faruqui et al 2015, NAACL best paper](https://www.aclweb.org/anthology/N15-1184/)

<center>
<img src="https://miro.medium.com/max/1280/0*QsBIIogEUALmMx8k.png" alt="image" height="200">
</center>


<center>
<img src="/assets/images/image_40_nlu_08.png" alt="image" width="450">
</center>


<center>
<img src="/assets/images/image_40_nlu_09.png" alt="image" width="400">
</center>


<center>
<img src="/assets/images/image_40_nlu_10.png" alt="image" width="400">
</center>

:paperclip: **Reference:**

- [Talk by author Manaal Faruqui](https://www.youtube.com/watch?v=yG4XbgytH4w&feature=youtu.be)
- [CS224U Youtube Lecture](https://www.youtube.com/watch?v=pip8h9vjTHY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=4)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Sentiment Analysis

## Core Reading

- [Sentiment Treebank - Richard Socher, EMNLP 2013](https://www.aclweb.org/anthology/D13-1170.pdf)
- [Opinion mining and sentiment analysis - Pang and Lee (2008)](https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)
- [A Primer on Neural Network Models for Natural Language Processing - Goldberg 2015](https://arxiv.org/abs/1510.00726)

## Introduction

`Sentiment analysis` seems simple at first but turns out to exhibit all of the complexity of full natural language understanding. To see this, consider how your intuitions about the sentiment of the following sentences can change depending on perspective, social relationships, tone of voice, and other aspects of the context of utterance:

1. There was an earthquake in LA.
2. The team failed the physical challenge. (We win/lose!)
3. They said it would be great. They were right/wrong.
4. Many consider the masterpiece bewildering, boring, slow-moving or annoying.
5. The party fat-cats are sipping their expensive, imported wines.
6. Oh, you're terrible!

## Related paper

- :book: [Subjectivity - Pang and Lee 2008](https://www.cs.cornell.edu/home/llee/papers/cutsent.pdf)
- :book: [Bias - Recasens et al. 2013](https://nlp.stanford.edu/pubs/neutrality.pdf)
- :book: [Stance - Anand et al. 2011](https://www.aclweb.org/anthology/W11-1701/)
- :book: [Abusive Language Detection - Nobata et al. 2016](https://pdfs.semanticscholar.org/e39b/586e561b36a3b71fa3d9ee7cb15c35d84203.pdf)
- :book: [Sarcasm - Khodak et al. 2017](https://arxiv.org/abs/1704.05579)
- :book: [Deception and Betrayal - Niculae et al. 2015](https://arxiv.org/abs/1506.04744)
- :book: [Online Troll - Cheng et al. CSCW 2017](https://arxiv.org/abs/1702.01119)
- :book: [Polarization - Gentzkow et al. 2019](https://arxiv.org/abs/1904.01596)
- :book: [Politeness - Danescu-Niculescu-Mizil et al. 2013](https://nlp.stanford.edu/pubs/politeness.pdf)
- :book: [Linguistic alignment - Doyle et al. 2013](https://www.aclweb.org/anthology/P16-1050.pdf)

_*Dataset available for these paper_

:paperclip: **Reference:**

- [CS224U Youtube Lecture](https://www.youtube.com/watch?v=O1Xh3H1uEYY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# What is Lexicon

A word to meaning dictionary

- [Slide Page 10](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-sentiment-handout.pdf)

# Sentiment Dataset

- [Slide Page 9](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-sentiment-handout.pdf)

----

# Art of Tokenization

Normal words are fine for tokenization. But it becomes problematic when you process social media post, e.g Tweet.

<center>
<img src="https://camo.githubusercontent.com/9117c63a6795f6981aaa42d7dc4f5640d31b1cd1/687474703a2f2f692e696d6775722e636f6d2f4b714a6e5654782e706e67" width="400">
</center>


![image](https://jenniferbakerconsulting.com/wp-content/uploads/JKD.jpg)

- `whitespace` tokenization is Okay. Basic. Preserve emoticons.
- `treebank` tokenizer - most systems are using.
  - Destroys hashtags, http link emoticons
- `sentiment aware` tokenization
  - `nltk.tokenize.casual.TweetTokenizer`
  - Isolate emoticon, respect domain specific tweet markup, capture multi-word expression, preserves capitalization where seems meaningful.

Different tokenization has impact on the final nlp task.  

:paperclip: **Reference:**

- [CS224U Youtube Lecture](https://www.youtube.com/watch?v=O1Xh3H1uEYY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# Danger of Stemming

- Stemming is heuristically collapsing words together by trimming of their ends. The idea is, this is helping you to collapse `morphological` variance.
- `porter` and `lancaster` destroy too many sentiment distinctions.
- `wordnet` is better but still not best.
- All comparative and superlative adjectives are stripped down to their base form
- `Sentiment-aware` tokenizer beats both `porter` and `lancaster`


## Question

**Does stemming work on misspelling words?**

**Ans:** Stemming is a set of rules and they are not intelligent. So whatever input you feed to them, if the rules are applicable, then they will be applied. However, for modern NLU model, misspelling words are not that much of a problem due to dense representations of words **generated from the context**. Therefore, if you have a common misspelling, then their word representation will be similar to the actual correct word. And this is one of the selling point of the modern NLU papers which state that no need of spell checker as preprocessing steps as the system will gracefully recover from that due to **context aware word vector representation**.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# POS tagging

POS tagging helps in improving sentiment classification task. Because there are words for which if multiple pos tags are available, then sentiment of different tags are different.

**Example:** 

- `fine`: if Adjective POSITIVE, if Noun then NEGATIVE


But even that Sentiment distinctions transcends (goes beyond) parts of speech.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Stanford Sentiment Treebank (SST)

- :page_with_curl: [Sentiment Treebank - Richard Socher, EMNLP 2013](https://www.aclweb.org/anthology/D13-1170.pdf)

Most sentiment prediction systems work just by looking at words in isolation, giving positive points for positive words and negative points for negative words and then summing up these points. **Problem:** the order of words is ignored and important information is lost. 

In constrast, our new deep learning model actually 

> **builds up a representation of whole sentences based on the `sentence structure`.**

**It computes the sentiment based on how words compose the meaning of longer phrases.**

<center>
<img src="https://miro.medium.com/max/1400/1*sh9P4hY6oR0mFbzWkUFtNA.png" height="300">
</center>

This way, the model is not as easily fooled as previous models. For example, our model learned that `funny` and `witty` are positive but the following sentence is still negative overall:

>> This movie was actually neither that funny, nor super witty.

The underlying technology of this demo is based on a new type of Recursive Neural Network that builds on top of grammatical structures. 


<center>
<img src="/assets/images/image_40_nlu_11.png" height="400">
</center>

Sentiment from bottom up is projected towards the top.

- :large_blue_circle: positive
- :white_circle: neutral
- :red_circle: negative

**SST-5** consists of $11855$ sentences extracted from movie reviews with fine-grained sentiment labels $[1–5]$, as well as $215154$ phrases that compose each sentence in the dataset.

The raw data with phrase-based fine-grained sentiment labels is in the form of a tree structure, designed to help train a **Recursive Neural Tensor Network** (**RNTN**) from their 2013 paper. The component phrases were constructed by parsing each sentence using the Stanford parser (section 3 in the paper) and creating a recursive tree structure as shown in the below image. A deep neural network was then trained on the tree structure of each sentence to classify the sentiment of each phrase to obtain a cumulative sentiment of the entire sentence.

:paperclip: **Reference:**

- [Stanford sentiment](https://nlp.stanford.edu/sentiment/)
- [Blog](https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4)
- Check the noebooks on sst from the course website

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Sentiment classifier comparison

<center>
<img src="/assets/images/image_40_nlu_12.png" height="300">
</center>

:paperclip: **Reference:**

- [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
- [McNemar test](https://en.wikipedia.org/wiki/McNemar%27s_test)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Assessing individual feature selection

<center>
<img src="/assets/images/image_40_nlu_13.png" width="400">
</center>

These kind of feature selection in the presence of `correlated features` are hard to interpret. So handle with care.


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Exercise:

- What is UMLFit ?
- What is U-Net

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
---
layout: post
title:  "Deep Learning: Natural Language Processing (Part 2)"
date:   2019-07-13 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

# High performance NLP

<object data="http://gabrielilharco.com/publications/EMNLP_2020_Tutorial__High_Performance_NLP.pdf" type="application/pdf" width="750px" height="750px">
    <embed src="http://gabrielilharco.com/publications/EMNLP_2020_Tutorial__High_Performance_NLP.pdf" type="application/pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://gabrielilharco.com/publications/EMNLP_2020_Tutorial__High_Performance_NLP.pdf">Download PDF</a>.</p>
    </embed>
</object>



----

# ELMo: Embeddings from Language Models

> :bulb:  This model was published early in 2018 and uses Recurrent Neural Networks (RNNs) in the form of Long Short Term Memory (LSTM) architecture to generate contextualized word embeddings

ELMo, unlike BERT and the USE, is not built on the transformer architecture. It uses LSTMs to process sequential text. ELMo is like a bridge between the previous approaches such as GLoVe and Word2Vec and the transformer approaches such as BERT.

Word2Vec approaches generated **static vector representations** or words which did not take `order` into account. There was one embedding for each word regardless of how it changed depending on the context, e.g. the word `right`, as in `it is a human right`, `take a right turn`, and  `that is the right answer`.


ELMo provided a significant step towards pre-training in the context of NLP. The ELMo LSTM would be trained on a massive dataset in the language of our dataset, and then we can use it as a component in other models that need to handle language.

**What’s ELMo’s secret?**

ELMo gained its language understanding from being trained to predict the next word in a sequence of words - a task called Language Modeling. This is convenient because we have vast amounts of text data that such a model can learn from without needing labels.

## Archiecture

ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass:

![image](https://blog.floydhub.com/content/images/2019/07/elmo.gif)

Unlike traditional word embeddings such as word2vec and GLoVe, the ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts.


**Reference:**

- [Learn ELMo for Extracting Features from Text](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# USE: Universal Sentence Encoder

> :bulb: The Universal Sentence Encoder (USE) was also published in 2018 and is different from ELMo in that it uses the Transformer architecture and not RNNs. This provides it with the capability to look at more context and thus generate embeddings for entire sentences.

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. 

The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.

## Architecture

Text will be tokenized by Penn Treebank(PTB) method and passing to either transformer architecture or deep averaging network. As both models are designed to be a general purpose, multi-task learning approach is adopted. The encoding model is designed to be as general  purpose  as  possible.   This  is  accomplishedby  using  multi-task  learning  whereby  a  single encoding  model  is  used  to  feed  multiple  down-stream tasks. The supported tasks include:

-  Same as Skip-though, predicting previous sentence and next sentence by giving current sentence.
-  Conversational response suggestion for the inclusion of parsed conversational data.
-  Classification task on supervised data

![image](https://blog.floydhub.com/content/images/2019/07/use.png)

> :bulb: The USE is trained on different tasks which are more suited to identifying sentence similarity.

Transformer architecture is developed by Google in 2017. It leverages self attention with multi blocks to learn the context aware word representation.

Deep averaging network (DAN) is using average of embeddings (word and bi-gram) and feeding to feedforward neural network.

![image](https://miro.medium.com/max/539/1*v07lrQnceNCWXxyVx2yixg.png)

The reasons of introducing two models because different concern. Transformer architecture achieve a better performance but it needs more resource to train. Although DAN does not perform as good as transformer architecture. The advantage of DAN is simple model and requiring less training resource.

## Why USE works better for sentence similarity task?

> :bulb: The USE is trained on a number of tasks but one of the main tasks is to identify the similarity between pairs of sentences. The authors note that the task was to identify “semantic textual similarity (STS) between sentence pairs scored by Pearson correlation with human judgments”. This would help explain why the USE is better at the similarity task.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

**Reference:**

- [Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf)
- [multi-task-learning-for-sentence-embeddings](https://medium.com/@makcedward/multi-task-learning-for-sentence-embeddings-55f47be1610a)
- [When Not to Choose the Best NLP Model](https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/) :fire:

----

# BERT: Bidirectional Encoder Representations from Transformers

> :bulb: BERT is the model that has generated most of the interest in deep learning NLP after its publication near the end of 2018. It uses the transformer architecture in addition to a number of different techniques to train the model, resulting in a model that performs at a SOTA level on a wide range of different tasks

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/BaPM47hO8p8" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/BaPM47hO8p8)_

## Architecture

BERT is trained on two main tasks:

:atom_symbol: **Masked language model:** Where some words are hidden (15% of words are masked) and the model is trained to predict the missing words

:atom_symbol: **Next sentence prediction:** Where the model is trained to identify whether sentence B follows (is related to) sentence A.


The new **XLNet** model improves on BERT since it uses the transformer XL, an extension of the transformer which enables it to deal with longer sentences than BERT.


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

**Reference:**

- [Jay Alammar illustrated-bert](https://jalammar.github.io/illustrated-bert/) :fire:

----

# XLNet

> :bulb: This is the newest contender to the throne of “Coolest New NLP Model". It uses a different approach than BERT to achieve bidirectional dependencies (i.e. being able to learn context by not just processing input sequentially). It also uses an extension of the transformer architecture known as Transformer XL, which enables longer-term dependencies than the original transformer architecture.

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
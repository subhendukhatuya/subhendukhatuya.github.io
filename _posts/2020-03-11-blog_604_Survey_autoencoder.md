---
layout: post
title:  "Survey - Auto-encoder"
date:   2020-03-11 00:00:10 -0030
categories: jekyll update
mathjax: true
---


# Content

1. TOC
{:toc}

----
# Introduction

According to this [blog](https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726):

Autoencoder is an **unsupervised artificial neural network** that learns how to efficiently **compress and encode** data then **learns how to reconstruct** the data back from the reduced encoded representation to a representation that is as close to the original input as possible.


> Autoencoder, by design, reduces data dimensions by learning how to ignore the noise in the data.


<center>
<img src="https://miro.medium.com/max/700/1*P7aFcjaMGLwzTvjW3sD-5Q.jpeg" width="500">
</center>

From architecture point of view, it looks like this:

<center>
<img src="https://miro.medium.com/max/1096/1*ZEvDcg1LP7xvrTSHt0B5-Q@2x.png" width="300">
</center>

# Implementation

According to this wonderful [notebook](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb) we can build the autoencoder as follows 

```py
##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        
        ### ENCODER
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary, 
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        
        ### DECODER
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        
        ### DECODER
        logits = self.linear_2(encoded)
        decoded = torch.sigmoid(logits)
        
        return decoded

    
torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

```

# Theory

Listen to the [lecture 7](https://www.cse.iitm.ac.in/~miteshk/CS7015.html) by Prof. Mitesh 

![image](/assets/images/image_37_ae_1.png)
![image](/assets/images/image_37_ae_2.png)
![image](/assets/images/image_37_ae_3.png)

## Choice of $f(x_i)$ and $g(x_i)$

![image](/assets/images/image_37_ae_4.png)
![image](/assets/images/image_37_ae_5.png)

## Choice of Loss Function

![image](/assets/images/image_37_ae_6.png)

_for mathematical proof check slide 11-12 of Lecture 7 by Prof. Mitesh_

![image](/assets/images/image_37_ae_7.png)

_for mathematical proof check slide 13-14 of Lecture 7 by Prof. Mitesh_


## Link between PCA and Autoencoders

![image](/assets/images/image_37_ae_8.png)

_for mathematical proof check slide 17-21 of Lecture 7 by Prof. Mitesh_

## Regularization in autoencoders

- The simplest solution is to add a **L2-regularization**  term  to  the  objective function

<center>

$
argmin_{\theta} \frac{1}{m} \sum\limits_{i=1}^m \sum\limits_{j=1}^n (\hat{x_{ij}} - x_{ij})^2 + \lambda \vert \vert \theta \vert \vert^2
$

</center>

- Another trick is to tie the weights of the  encoder and decoder i.e. , $W^*=W^T$.
- This  effectively  reduces  the  capacity of Autoencoder and acts as a regularizer.

![image](/assets/images/image_37_ae_11.png)

- Different regularization gives rise to different Autoencoder as seen in the above slide. The second and third regularization gives rise to **Spare AE** and **Contractive AE**. For more details check the slide. 

## Denoising Autoencoder

![image](/assets/images/image_37_ae_9.png)
![image](/assets/images/image_37_ae_10.png)


**Reference:**

- [Very Important Lecture 7, both slide and video](https://www.cse.iitm.ac.in/~miteshk/CS7015.html)

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>

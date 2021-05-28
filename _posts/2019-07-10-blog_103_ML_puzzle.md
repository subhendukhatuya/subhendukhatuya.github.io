---
layout: post
title:  "Machine Learning Puzzles"
date:   2019-07-10 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content 

1. TOC
{:toc}
---

Interesting ML Puzzle (curated over internet)

# [ML Questions] Overfitting: 

I am training a Random Forest on 1 million ($10^6$) points having 10000 ($10^4$) dimensions. I have already trained 5000 trees and want to train another 10000. Should I go ahead and train 15000 trees or do I have danger of overfitting? [[Qlink](https://www.linkedin.com/feed/update/urn:li:activity:6498386172857933824/)]

- **[Ans]**: In his 1999 seminal paper (Theorem 1.2), Breiman already mathematically proved that Random Forest does not overfit on the number of trees. This is due to the `law of large numbers`. The error will always converge with more number of trees. [[paper link](https://www.stat.berkeley.edu/~breiman/random-forests.pdf)]
- **[Q]**:  How many features (apprx) are used at a time to train single tree? 
  - $\log_e(K)$, where K is the dimension of the input data, i.e number of features.
- **[Observation]**: With more trees the training and prediction time will be higher. Why?
  - Prediction time is usually a big factor. Just increasing the number of trees will  soon become prohibitive. In my case, assume every tree was balanced, then I will need ($\log_2(10^6)$) 20 iteration on average to reach to leaf. With 10000 trees, I am doing 20 X 10000 threshold evaluations at prediction time. Very expensive.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# [ML Questions] Bias and Variance: 

Assume that I am training a random forest, where each tree is grown fully. The training data consists of N samples. To train a tree I create a subset of size N by sampling with replacement from training data.
The original training data is composed of F features, and for determining the split at any node in a tree, I am using $\log_e(F)$ features as candidates.
Assume that the trees are grown fully.
If we consider the individual trees, will they have `high variance, high bias`, or nothing can be said about individual trees?
When we combine the trees, are we trying to correct the variance or the bias or both? [[Qlink](https://www.linkedin.com/feed/update/urn:li:activity:6497416642216198144/)]

- [[effect of different hyper-parameter](https://towardsdatascience.com/random-forests-and-the-bias-variance-tradeoff-3b77fee339b4)]
-  A single decision tree if fully grown that means you are increasing the complexity of the tree. This implies the bias is reduced but variance increased.
- **[My Ans]**: here even if each tree is grown fully, they are grown over a dataset of same size of original data, but sampled with replacement. So they haven't seen the full data, so they shouldn't overfit i.e variance will be low. But w.r.t each sampled dataset each tree is fully grown, so they have high variance and low bias (fully grown). By combining the trees we are correcting the variance. 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----


# [ML Question] Search Engine

Assume that we have a corpus of `1 billion` images. I also have a black box function `F()` that can calculate the similarity between two images in `1 millisecond`. We are `not allowed` to extract any features based on the images nor we can reverse engineer the function F().

Given the above setup, we are supposed to create a search engine such that the user can supply an input image, which may not be present in our corpus, and we are supposed to return the top 10 similar images based on F().

How will you design such a search engine ?

[Q_link](https://www.linkedin.com/posts/arahul_mlquestion-similarityfunction-ml-activity-6564546445360361472-r1YS)

- [**My Ans:**] Apply K-Means clustering first. Then check (measure similarity with `F()`) the incoming image from user to the central image of each cluster and pick the cluster with closest match. And then inside that cluster match with other for picking top 10.
- [**Other Answer**] The basic idea here has to have some form of clustering, the only challenge here being that we do not have direct representations to cluster on. But, we know that k-means is in a way a linear algorithm and hence has a kernel version of it as well. Basically we can represent the same k-means algorithm in a kernelized fashion (similar to kernel lin-regression). [paper](https://dl.acm.org/citation.cfm?id=1014118)
  - [Parsimonious Online Learning with Kernels via Sparse Projections in Function Space](https://arxiv.org/abs/1612.04111)
  - [slide from Prof. Piyush Rai, IIT Kanpur](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec10_slides.pdf)
  - [Prof. Piyush Rai ML Course](https://www.cse.iitk.ac.in/users/piyush/courses/ml_autumn16/ML.html)

- It can be solved using perpetual image hashing techniques and Locality Sensitive Hashing. 
  - [ImageHash](https://github.com/JohannesBuchner/imagehash)
  - [DataSketch](https://github.com/ekzhu/datasketch)

- Randomly select N images (representing mean images of N clusters), then use F() to assign each image to a cluster. 
To compute new mean of each cluster (assume C1), compute the similarity between every image of C1. For each image sum the similarity values with other images of C1 and select the one with highest value as the new mean. (Intuition - image which is most similar to the other images in the cluster is the mean.)
Continue the K-means process to find the final clusters and means. 
Then compare each "mean" image with input image to find the most similar cluster. After selecting the cluster, check with each image of that cluster the similarity value and output the 10 images with highest similarity values.
  - If $N$ (number of clusters) = $10^5$.
Each cluster with have approximately $10^4$ images.
Worst case time = $(10^5 + 10^4) * 10^{-3} = 110$ sec. 

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
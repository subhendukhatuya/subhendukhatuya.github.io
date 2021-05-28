---
layout: post
title:  "Probabilistic and Bayesian Machine Learning"
date:   2019-09-07 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

# Probabilistic Machine Learning

- [Prof. Piyush Rai, IIT Kanpur, Probabilistic Machine Learning](https://www.cse.iitk.ac.in/users/piyush/courses/pml_winter16/PML.html)

---

# Bayesian Machine Learning

- [Prof. Piyush Rai, IIT Kanpur, Bayesian Machine Learning](https://www.cse.iitk.ac.in/users/piyush/courses/bml_winter17/bayesian_ml.html)

----

# Bayesian Network Learning

![image](/assets/images/image_02_bnlearn_1.png)
![image](/assets/images/image_02_bnlearn_2.png)
![image](/assets/images/image_02_bnlearn_2_1.png)
![image](/assets/images/image_02_bnlearn_3.png)
![image](/assets/images/image_02_bnlearn_4.png)
![image](/assets/images/image_02_bnlearn_4_1.png)
![image](/assets/images/image_02_bnlearn_5.png)
![image](/assets/images/image_02_bnlearn_6.png)

----

## Structure learning for bayesian networks

According to this [blog](https://ermongroup.github.io/cs228-notes/learning/structure/)


The task of structure learning for Bayesian networks refers to learn the structure of the directed acyclic graph (DAG) from data. There are two major approaches for the structure learning: score-based approach and constraint-based approach .

**Score-based approach**

- The score-based approach first defines a criterion to evaluate how well the Bayesian network fits the data, then searches over the space of DAGs for a structure with maximal score.
- In this way, **the score-based approach is essentially a search problem** and consists of two parts: the definition of score metric and the search algorithm.

Score metrics

The score metrics for a structure $\mathcal{G}$ and data $D$ can be generally defined as:

<center>

$
Score(G:D)= LL(G:D) - \phi(\vert D \vert) \vert \vert G \vert \vert.
$

</center>

Here $LL(G:D)$ refers to the **log-likelihood of the data** under the graph structure $\mathcal{G}$. The parameters in Bayesian network $G$ are estimated based on MLE and the log-likelihood score is calculated based on the estimated parameters. If we consider only the log-likelihood in the score function, we will end up with an overfitting structure (namely, a complete graph.) That is why we have the second term in the scoring function. $\vert D \vert$ is the number of sample and $\vert \vert G \vert \vert$ is the number of parameters in the graph $\mathcal{G}$. With this extra term, we will penalize the over-complicated graph structure and avoid overfitting. 


**Search algorithms**

The most common choice for search algorithms are local search and greedy search.

For local search algorithm, it starts with an empty graph or a complete graph. At each step, it attempts to change the graph structure by a single operation of adding an edge, removing an edge or reversing an edge. (Of course, the operation should preserve the acyclic property.) If the score increases, then it adopts the attempt and does the change, otherwise it makes another attempt.

## Implementation

```r
> library(bnlearn)
> data(learning.test)
> pdag = iamb(learning.test)
> pdag


#  Bayesian network learned via Constraint-based methods

#  model:
#    [partially directed graph]
#  nodes:                                 6 
#  arcs:                                  5 
#    undirected arcs:                     1 
#    directed arcs:                       4 
#  average markov blanket size:           2.33 
#  average neighbourhood size:            1.67 
#  average branching factor:              0.67 

#  learning algorithm:                    IAMB 
#  conditional independence test:         Mutual Information (disc.) 
#  alpha threshold:                       0.05 
#  tests used in the learning procedure:  134

```

As we can see from the output above, there is a single undirected arc in pdag; `IAMB` was not able to set its orientation because its two possible direction are score equivalent.


```r
> score(set.arc(pdag, from = "A", to = "B"), learning.test)
# [1] -24006.73

> score(set.arc(pdag, from = "B", to = "A"), learning.test)
# [1] -24006.73
```

## Fitting the parameters (Maximum Likelihood estimates)

Discrete data

Now that the Bayesian network structure is completely directed, we can fit the parameters of the local distributions, which take the form of conditional probability tables.

```r
> fit = bn.fit(dag, learning.test)
> fit


#  Bayesian network parameters

#  Parameters of node A (multinomial distribution)

#Conditional probability table:
#     a     b     c 
# 0.334 0.334 0.332 

#  Parameters of node B (multinomial distribution)

#Conditional probability table:
 
#   A
# B        a      b      c
#  a 0.8561 0.4449 0.1149
#  b 0.0252 0.2210 0.0945
#  c 0.1187 0.3341 0.7906

# ...
```


**Reference:**

- [Oxford: slide 1](https://www.bnlearn.com/about/slides/slides-aist17.pdf)
- [Oxford: slide 1](https://www.bnlearn.com/about/teaching/slides-bnshort.pdf)
- [MIT: Slide](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-951j-medical-decision-support-spring-2003/lecture-notes/lecture6.pdf)
- [Book: Learning Bayesian Network](http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf)
- [Columbia: Slide](https://www.ee.columbia.edu/~vittorio/Lecture12.pdf)
- [Imp Blog](https://ermongroup.github.io/cs228-notes/learning/structure/)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>


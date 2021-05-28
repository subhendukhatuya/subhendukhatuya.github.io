---
layout: post
title:  "Machine Learning Concepts (Part 1)"
date:   2019-07-04 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

# Linear Regression Assumption

There are four assumptions associated with a linear regression model:

- **Linearity:** The relationship between $X$ and the mean of $Y$ is linear.
- **Homoscedasticity:** The variance of `residual` ($(Y - \hat Y)$) is the same for any value of $X$.
- **Independence:** Observations are independent of each other.
- :fire: **Normality:** For any fixed value of $X$, $Y$ is normally distributed. The error($\epsilon = (Y_i - \hat{Y_i})$ residuals) follow a normal distribution $\mathcal{N}(\mu , \sigma)$


**Reference:**

- [Simple Linear Regression](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/R/R5_Correlation-Regression/R5_Correlation-Regression4.html)
- [Assumptions Of Linear Regression Algorithm](https://towardsdatascience.com/assumptions-of-linear-regression-algorithm-ed9ea32224e1) :fire:

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Categorization of ML Algorithms

To aid our conceptual understsanding, each of the algorithms can be categorized into various categories:

- eager vs lazy
- batch vs online
- parametric vs nonparametric
- discriminative vs generative

:atom_symbol: **Eager vs lazy learners:** Eager learners are algorithms that process training data `immediately`  whereas  lazy  learners  defer  the  processing  step  `until  the  prediction`.   In  fact,  lazy learners do not have an explicit training step other than storing the training data.  A popular example of a lazy learner is the **Nearest Neighbor algorithm**.

:atom_symbol: **Batch  vs  online  learning:** Batch  learning  refers  to  the  fact  that  the  model  is  learned on the `entire set` of training examples.  Online learners, in contrast, learn from `one training example` at the time.  It is not uncommon, in practical applications, to learn a model via batch learning and then update it later using online learning.


:atom_symbol: **Parametric vs nonparametric models:** Parametric models are `fixed` models, where we assume a certain functional form for $f(x) =y$ .  For example,  linear regression can be considered  as  a  parametric  model  with $h(x)  = w_1 x_1+ \dots + w_m x_m+b$ .   Nonparametric models are more `flexible` and do not have a pre-specfied number of parameters.  In fact,the number of parameters grows typically with the size of the training set.  For example, a **decision tree** would be an example of a nonparametric model, where each decision node(e.g., a binary `True/False` assertion) can be regarded as a parameter.

    
:atom_symbol: **Discriminative  vs  generative:** Generative  models  (classically)  describe  methods  that model  the  `joint  distribution` $P(X,Y)  = P(Y)P(X \vert Y)  =P(X)P(Y \vert X)$  for  training  pairs $(x_i,y_i)$. Discriminative models are taking a more `direct` approach, modeling $P(Y \vert X)$ directly.  While generative models provide typically more insights and allow sampling from the  joint  distribution,  discriminative  models  are  typically  easier  to  compute  and  produce more accurate predictions.  Helpful for understanding discriminative models is the following analogy:  

> :bulb: Discriminative modeling is like trying to extract information $Y$ `given` (from) text $X$ in a foreign language ($P(Y \vert X)$) **without learning that language**.

**Reference:**

- [stat451-machine-learning-fs20 - Sebastian Rachka - Lecture 1](https://github.com/rasbt/stat451-machine-learning-fs20) :fire:

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Algorithm selection Tips 

- When to use and when not to use.
- Merits and Demerits
- Pros and Cons

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to Prepare for ML-DL Interview

> Interviews are stressful, even more so when they are for your dream job. As someone who has been in your shoes, and might again be in your shoes in the future, I just want to tell you that it doesn’t have to be so bad. Each interview is a learning experience. An offer is great, but a rejection isn’t necessarily a bad thing, and is never the end of the world. I was pretty much rejected for every job when I began this process. Now I just get rejected at a less frequent rate. Keep on learning and improving. You’ve got this!

- [Tweet: Chip Huyen](https://twitter.com/chipro/status/1152077188985835521)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

---

# Difference between Parametric and Non-Parametric model?

:large_orange_diamond: **Parametric Model:** We have a finite number of parameters. And once you have determined the right parameters, you can throw away the training data. The prediction is based on the learnt weight only.

> :bulb: ... prediction is independent of the training data ... 

Linear models such:

- Linear regression
- Logistic regression
- Linear Support Vector Machines 
- Mixture Models, K-Means
- HMM
- Factor Analysis

These are typical examples of a `parametric learners`.

:large_orange_diamond: **Nonparametric Model:** The number of parameters is (potentially) infinite. Here number of parameters are function of number of training data.

> :bulb: ... prediction is dependent of the training data ...

> Non-parametric  models assume  that  the  data  distribution  cannot  be  defined  in terms  of  such  a  finite  set  of  parameters. Model parameters actually grows with the training set.  You can imagine each training instance as a "parameter" in the model, because they're the things you use during prediction

- K-nearest neighbor - all the training data are needed during prediction.
- Decision trees - All the training data needed during prediction. The test input will pass through each decision tree (each decision tree is nothing but some automatic rules based on the full training data) and finally some voting to get the actual prediction.
- RBF kernel SVMs 
- Gaussian Process
- Dirichlet Process Mixture
- infinite HMMs
- Infinite Latent Factor Model

These are considered as `non-parametric`

- In the field of statistics, the term parametric is also associated with a specified probability distribution that you **assume** your data follows, and this distribution comes with the finite number of parameters (for example, the mean and standard deviation of a normal distribution).
- You don’t make/have these assumptions in non-parametric models. So, in intuitive terms, we can think of a non-parametric model as a `“distribution” or (quasi) assumption-free model`.


![image](/assets/images/image_04_param_non_param_1.png)

**Reference:**

- [MLSS2015 Tuebingen, slide](http://mlss.tuebingen.mpg.de/2015/slides/ghahramani/gp-neural-nets15.pdf)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Linear Regression

- It's linear with respect to weight `w`, but not with respect to input `x`.
- `posterior ~ likelihood * prior`
- Ordinary Least Square (OLS) approach to find the model parameters is a 
  - **special case of maximum likelihood estimation** 
  - The overfitting problem is a general property of the MLE. But by adopting the Bayesian approach, the overfitting problem can be avoided. `[p9, Bishop]`
- Also from Bayesian perspective we can use model for which number of parameters can exceed the number of training data. In Bayesian learning, the effective number of parameters adapts automatically to the size of the data.

## Point Estimate of W vs W Distribution

Consider $D$ is our dataset and $w$ is the parameter set. Now in both Bayesian and frequentist paradigm, the likelihood function $P(D \vert w)$ plays a central role. In **frequentist approach**, $w$ is considered to be `fixed parameter`, whose value is determined by some form of estimator and the error bars on the estimator are obtained by considering the distribution of possible data sets $D$. 

However, in **Bayesian setting** we have only one single datasets $D$ (the observed datasets), and the uncertainty in parameters is expressed through a probability distribution over $w$. `[p22-p23]`

:rocket: **A widely used frequentist estimator is the maximum likelihood**, in which $w$ is set to the value that maximizes $P(D \vert w)$. In the ML literature, the `negative log of the likelihood` is considered as _error function_. As the negative log is a monotonically decreasing function, so `maximizing the likelihood is equivalent to minimizing the error`.

However Bayesian approach has a very common criticism, i.e. the inclusion of prior belief. Therefore, people try to use _noninformative prior_ to get rid of the prior dependencies. `[p23]`.

## Criticism for MLE

- MLE suffers from `overfitting` problem. In general overfitting means the model fitted to the training data so perfectly that if we slightly change the data and get prediction, test error will be very high. That means the model is **sensitive to the variance of data**. 
- The theoretical reason is MLE systematically undermines the variance of the distribution. See proof `[[p27], [figure 1.15,p26, p28]]`.  Because here `sample variance` is measured using the `sample mean`, instead of the population mean.
- Sample mean is an unbiased estimator of the population mean, but sample variance is a biased estimator of the population variance. `[p27], [figure 1.15,p26]`
- If you see the image `p28`, the 3 red curve shows 3 different dataset and the green curve shows the true dataset. And the mean of the 3 red curves coincide with the mean of the green curve, but their variances are different. 

In Bayesian curve fitting setting, the sum of squared error function has arisen as a consequence of maximizing the likelihood under the assumption of Gaussian noise distribution.

Regularization allows complex models to be trained on the datasets of limited size without severe overfitting, essentially by limiting the model complexity. `[p145, bishop]`

>> :sparkles: **In bayesian setting, the prior works as the Regularizer.**

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Bias Variance Trade-off from Frequentist viewpoint

The goal to minimize a model's generalization error gives rise to two desiderata:

1. Minimize the training error; 
2. Minimize the gap between training and test error.

:radio_button: **Minimize the training error - Reducing Bias:**
This dichotomy is also known as bias-variance trade-off. If the model is not able to obtain a low error on the training set, it is said to have `high bias`. This is typically the result of `erroneous assumptions in the learning algorithm` that cause it to miss relevant relations in the data. 

:radio_button: **Minimize the gap between training and test error - Reducing Variance:**
On the other hand, if the gap between the training error and test error is too large, the model has high variance. It is sensitive to small fluctuations and models random noise in the training data rather than the true underlying distribution.


In frequentist viewpoint, `w` is fixed and error bars on the estimators are obtained by considering the distribution over the data $D$. `[p22-23; Bishop]`.

Suppose we have large number of **data sets**, `[D1,...,Dn]`, each of size N and each drawn independently from distribution of `P(t,x)`. For any given data set `Di` we can run our learning algorithm and get a prediction function `y(x;Di)`. Different datasets from the ensemble will give different prediction functions and consequently different values of squared loss. The performance of a particular learning algorithm is then assessed by averaging over this ensemble of datasets. `[p148; Bishop]`. 

Our original regression function is `Y` and say for `Di` we got our predictive function $\hat{Y_i}$.

:large_orange_diamond: **Bias** = $(E[\hat{Y_i}(x;D_i)] - Y)^2$, where $E[\hat{Y_i}(x;D_i)]$ is average (expected) performance over all the datasets. So, Bias represents the extent to which the average prediction over all the datasets $D_i$ differ from the desired regression function $Y$.
- :zap: **TL;DR:**  The  bias  of  an  estimator $\hat \beta$ is the **difference between its expected value** $E[\hat \beta]$ **and the true value** of a parameter $\beta$ being estimated: $Bias = E[\hat \beta] − \beta$

:large_orange_diamond: **Variance** = $E[(\hat{Y_i}(x;D_i) - E[\hat{Y_i}(x;D_i)])^2]$, where $(\hat{Y_i}(x;D_i)$ is the predictive function over data set $D_i$ and $E[\hat{Y_i}(x;D_i)])$ is the average performance over all the datasets. So variance represents, the extent to which the individual predictive functions $\hat{Y_i}$ for dataset $D_i$ varies around their average. And thus we measure the extent by which the function $Y(x;D)$ is sensitive to the particular choice of the data sets. `[p149; Bishop]`
- :zap: **TL;DR:** The variance is simply the statistical variance of the estimator $\hat \beta$ and its expected value $E[\hat \beta]$ , i.e $Var = E [(\hat{\beta} - E[\hat \beta])^2]$

_*Note: here patameter $\beta$ and estimator $\hat \beta$._

## Bias Variance Decomposition

![image](/assets/images/image_16_ModelSelection_4.png)


**Resource:**

- [Prof. Piyush Rai, Lecture 19](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec19_slides.pdf)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Linear Models

## General Linear Model

Indeed, the general linear model can be seen as an **extension of linear multiple regression for a single dependent variable**. Understanding the multiple regression model is fundamental to understand the general linear model.

## Single Regression
One independent variable and one dependent variable

<center>

$y=\theta_0+\theta_1 x_1$

</center>



## Multiple Regression

Multiple regression is an extension of simple linear regression. It is used when we want to predict the value of a variable based on the value of two or more other variables. The variable we want to predict is called the dependent variable (or sometimes, the outcome, target, response or criterion variable).

:zap: **TL;DR:** Single dependent/target/response variable and multiple ($\gt 1$) predictors/independent variables.

Multiple linear regression is the most common form of linear regression analysis.  As a predictive analysis, the multiple linear regression is used to explain the relationship between one continuous dependent variable and two or more independent variables

<center>

$y=\theta_0+\theta_1 x_1+\theta_2 x_2+\dots+\theta_n x_n = \theta_0 + \sum\limits_{i=1}^{n} \theta_i x_i$

</center>


## Additive Model:

<center>

$y=\theta_0+\theta_1 f_1(x_1)+\theta_2 f_2(x_2)+\dots+\theta_n f_n(x_n)= \theta_0 + \sum\limits_{i=1}^{n} \theta_i f(x_i)$

</center>

A generalization of the multiple regression model would be to maintain the additive nature of the model, but to replace the simple terms of the linear equation $\theta_i * x_i$ with $\theta_i * f_i(x_i)$ where $f_i()$ is a **non-parametric function** of the predictor $x_i$.  

In other words, instead of a single coefficient for each variable (additive term) in the model, in additive models an unspecified (non-parametric) function is estimated for each predictor, to achieve the best prediction of the dependent variable values.


## General Linear Model - Revisited

One way in which the `general linear model` differs from the `multiple regression model` is in terms of the number of dependent variables that can be analyzed. The $Y$ vector of $n$ observations of a single $Y$ variable can be replaced by a $\mathbf{Y}$ matrix of $n$ observations of $m$ different $Y$ variables. Similarly, the $w$ vector of regression coefficients for a single $Y$ variable can be replaced by a $\mathbf{W}$ matrix of regression coefficients, with one vector of $w$ coefficients for each of the m dependent variables. These substitutions yield what is sometimes called the multivariate regression model,

<center>

${\mathbf{Y}}=\mathbf{XW}+{E}$

</center>

where 

- $\mathbf{Y}$ is a matrix with series of **multivariate measurements** (each column being a set of measurements on one of the dependent variables)
- $\mathbf{X}$ is a matrix of observations on independent variables that might be a **design matrix** (each column being a set of observations on one of the independent variables).
- $\mathbf{W}$ is a matrix containing **parameters** that are usually to be estimated and $E$ is a matrix containing errors (noise). 

The errors are usually assumed to be uncorrelated across measurements, and follow a multivariate normal distribution. If the errors do not follow a multivariate normal distribution, generalized linear models may be used to relax assumptions about $\mathbf{Y}$ and $\mathbf{W}$.

## Generalized Linear Model

To summarize the basic idea, the `generalized linear model` differs from the `general linear model` (of which multiple regression is a special case) in two major respects: 
1. The distribution of the `dependent` or `response variable` _can be (explicitly) non-normal_, and does not have to be continuous, e.g., it can be binomial; 
2. The dependent (target or response) variable values are predicted from a linear combination of predictor variables, which are "connected" to the dependent variable via a `link function`.

:diamond_shape_with_a_dot_inside: **General Linear Model**

<center>

$y=\theta_0+\theta_1 x_1+\theta_2 x_2+...+\theta_n x_n=\theta_0 + \sum_i \theta_i x_i$

</center>

:diamond_shape_with_a_dot_inside: **Generalized Linear Model**

<center>

$y=g(\theta_0+\theta_1 x_1+\theta_2 x_2+...+\theta_n x_n)= g(\theta_0 + \sum_i \theta_i x_i)$

</center>

<center>

$g^{-i}(y)=\theta_0 + \sum_i \theta_i x_i$

</center>

where $g^{-i}()$ is the inverse of $g()$

### Link Function:

Inverse of $g()$, say $g^{-i}()$ is the `link function`.

## Generalized Additive Model:

We can combine the notion of `additive models` with `generalized linear models`, to derive the notion of `generalized additive models`, as:

<center>

$y = g(\theta_0 + \sum_{i=1}^n \theta_i f_i(x_i))$

</center>

<center>

$g^{-i}(y)=\theta_0+\theta_1 f_1(x_1)+\theta_2 f_2(x_2)+...+\theta_n f_n(x_n)= \theta_0 + \sum_{i=1}^n \theta_i f_i(x_i)$

</center>


**Reference:**

- [Link](http://www.statsoft.com/Textbook/Generalized-Additive-Models)


## Why do we call it GLM when it's clearly non-linear?


:sparkles: **Linear Model:** They are linear in parameters $\theta$

<center>

  $y_i = \theta_0 + \sum_{i=1}^{n} \theta_i x_i$

</center>

:sparkles: **Non Linear Model:** They are non-linear in parameters $\theta$

<center>

  $y_i = \theta_0 + \sum_{i=1}^{n} g(\theta_i) x_i$
  
</center>

where $g()$ is any non linear function.

**GLM:**

>> "..**`inherently nonlinear`** (they are nonlinear in the parameters), yet **`transformably linear`***, and thus fall under the GLM framework.."

Now GLM in `non-linear` due to the presence of $g()$ but it can be transformed into `linear in parameters` using `link function` $g^{-i}()$.

**Reference:**

- [Stackexchange](https://stats.stackexchange.com/questions/120047/nonlinear-vs-generalized-linear-model-how-do-you-refer-to-logistic-poisson-e)
  
<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# K-means vs kNN

The two most commonly used algorithms in machine learning are K-means clustering and k-nearest neighbors algorithm.

Often those two are confused with each other due to the presence of the k letter, but in reality, those algorithms are slightly different from each other.

Thus, K-means `clustering` represents an unsupervised algorithm, mainly used for clustering, while KNN is a `supervised learning` algorithm used for classification. 


----

# Nearest Neighbour

Nearest  neighbor  algorithms  are  among  the  `simplest`  supervised  machine  learning  algorithms and have been well studied in the field of pattern recognition over the last century.

:atom_symbol: While kNN is a `universal function approximator` under certain conditions, the underlying concept is relatively simple. kNN is an algorithm for supervised learning that simply stores the labeled training examples during the training phase.  For this reason, kNN is also called a `lazy learning algorithm`. What it means to be a lazy learning algorithm is that the processing of the training examples is postponed until making predictions once again, the training consists literally of just storing the training data.

> :bulb: When you are reading recent literature, note that the prediction step is now often called ”inference” in the machine learning community.  Note that in the context of machine learning, the term ”inference” is not to be confused with how we use the term ”inference” in statistics, though.

Then, to make a prediction (class label or continuous target), a trained kNN model finds the k-nearest  neighbors  of  a  query  point  and  computes  the  class  label  (classification)  or continuous target (regression) based on the k nearest (most `similar`) points.  The exact mechanics will be explained in the next sections.  However, the overall idea is that instead of approximating the target function $f(x) = y$ globally, during each prediction, kNN approximates the target function locally. 

Since kNN does not have an explicit training step and defers all of the computation until prediction, we already determined that kNN is a lazy algorithm. 

Further, instead of devising one global model or approximation of the target function, for each  different  data  point,  there  is  a  different  local  approximation,  which  depends  on  the data  point  itself  as  well  as  the  training  data  points.   Since  the  prediction  is  based  on  a comparison  of  a  `query  point`  with  `data  points`  in  the  training  set  (rather  than  a  globalmodel), kNN is also categorized as `instance-based` (or `memory-based`) method.  While kNN  is  a  **lazy  instance-based**  learning  algorithm,  an  example  of  an  eager  instance-based learning algorithm would be the SVM. 

Lastly,  because  we  do  not  make  any  assumption  about  the  functional  form  of  the kNN algorithm, a kNN model is also considered a **non-parametric** model.

## Curse of Dimensionality

The kNN algorithm is particularly susceptible to the curse of dimensionality.  In machine learning, the curse of dimensionality refers to scenarios with a fixed number of training examples but an increasing number of dimensions and range of feature values in each dimensionin a high-dimensional feature space. In kNN an increasing number of dimensions becomes increasingly problematic because the more dimensions we add, the larger the volume in the hyperspace needs to capture fixed number of neighbors.  As the volume grows larger and larger, the `neighbors` become less and less `similar` to the query point as they are now all relatively distant from the query point considering all different dimensions that are included when computing the pairwise distances.

## Big-O of kNN

For  the  brute force  neighbor  search  of  the kNN  algorithm,  we  have  a  time  complexity  of $O(n×m)$, where $n$ is the number of training examples and $m$ is the number of dimensions in the training set.  For simplicity, assuming $n \gggtr m$ , the complexity of the brute-force nearest-neighbor  search  is $O(n)$.   In  the  next  section,  we  will  briefly  go  over  a  few  strategies  to improve the runtime of the kNN model.

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

## Different data-structure of kNN:

- Heap
- **Bucketing:**
  - The  simplest  approach  is  `bucketing`.   Here,  we  divide  the  search  space  into  identical, similarly-sized  cells  (or  buckets),  that  resemble  a  grid  (picture  a  2D  grid  2-dimensional hyperspace or plane)
- **KD-Tree:**
  - A  KD-Tree,  which  stands  for `k-dimensional  search  tree`,  is  a  generalization  of  binary search trees.  KD-Trees data structures have a time complexity of $O(\log(n))$ on average (but $O(n)$  in  the  worst  case)  or  better  and  work  well  in  relatively  low  dimensions.   KD-Trees also partition the search space perpendicular to the feature axes in a Cartesian coordinate system.  However, with a large number of features, KD-Trees become increasingly inefficient,and alternative data structures, such as Ball-Trees, should be considered.
- **Ball Tree:**
  - In contrast to the KD-Tree approach, the Ball-Tree partitioning algorithms are based on the construction of `hyper spheres` instead of cubes.  While Ball-Tree algorithms are generally more expensive to run than KD-Trees, the algorithms address some of the shortcomings of the KD-Tree data structure and are more efficient in higher dimensions.

## Parallelizing kNN

kNN is one of these algorithms that are very easy to parallelize.  There are many different ways to do that. For instance, we could use distributed approaches like map-reduce and place subsets of the training datasets on different machines for the distance computations. Further,the  distance  computations  themselves  can  be  carried  out  using  parallel  computations  on multiple processors via CPUs or GPUs.

**Reference:** For more details must read the below reference:

- [stat451-machine-learning-fs20 - Sebastian Rachka - Lecture 2](https://github.com/rasbt/stat451-machine-learning-fs20) :fire:


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Eigen Decomposition

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/PFDu9oVAE-g" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/PFDu9oVAE-g)_


- [Application](https://www.intmath.com/matrices-determinants/8-applications-eigenvalues-eigenvectors.php)
  - Google's PageRank - Their task was to find the "most important" page for a particular search query from the link matrix
  - Repeated applications of a matrix: Markov processes - First, we need to consider the conditions under which we'll have a steady state. If there is no change of value from one month to the next, then the eigenvalue should have value $1$
  - ...

Matrices acts as **linear transformations**. Some matrices will **rotate** your space, others will **rescale** it etc. So when we apply a matrix to a vector, we end up with a transformed version of the vector. When we say that we `apply` the matrix to the vector it means that we calculate the dot product of the matrix with the vector. 


:dart: **Eigenvector:** Now imagine that the transformation of the initial vector gives us a new vector that has the exact same direction. **The scale can be different but the direction is the same**. Applying the matrix didn’t change the direction of the vector. Therefore, this type of initial vector is special and called an eigenvector of the matrix.


We can decompose the matrix $\mathbf{A}$ with eigenvectors and eigenvalues. It is done with: $\mathbf{A}=V* \Sigma * V^{−1}$ , where $\Sigma = diag(\lambda)$ and each column of $V$ is the eigenvector of $A$.


## Real symmetric matrix:

In the case of real symmetric matrices, the eigen decomposition can be expressed as
$\mathbf{A}=Q* \Sigma *Q^T$

where Q is the matrix with eigenvectors as columns and $\Sigma$ is $diag(\lambda)$.

+ Also $Q$ is an `orthonormal` matrix. So, $Q^TQ=I$ i.e. $Q^T=Q^{-1}$

## Why Eigenvalue and Eigenvectors are so important?

- They are quite helpful for optimization algorithm or more clearly in constrained optimization problem. In optimization problem we are solving a system of linear equation. Typical `Gaussian Elimination Technique` has time complexity $O(n^3)$ but this can be solved with Eigenvalue Decomposition which needs $O(n^2)$. So they are more efficient. [for more explanation follow the deep learning lecture 6 of prof. Mitesh from IIT Madrass] 

:bookmark_tabs: **References:**

- [Lecture 6: Prof.Mitesh_IIT-M](https://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture6.pdf)
- [(Book_IMPORTANT_link)](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.7-Eigendecomposition/)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Matrix Factorization

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/ZspR5PZemcs" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/ZspR5PZemcs)_


Matrix factorization is a simple **embedding model**. Given the feedback matrix $A \in \mathbb{R}^{m \times n}$ , where $m$ is the number of users (or queries) and $n$ is the number of items, the model learns:

- A **user embedding matrix** $U \in \mathbb{R}^{m \times d}$, where row $i$ is the embedding for user $i$.
- An **item embedding matrix** $V \in \mathbb{R}^{n \times d}$, where row $j$ is the embedding for item $j$.

![image](https://developers.google.com/machine-learning/recommendation/images/Matrixfactor.svg)

The embeddings are learned such that the product $UV^T$ is a **good approximation** of the `feedback matrix` $A$. Observe that the  $(i j)$ entry of $(U.V^T)$ is simply the **dot product** of the embeddings of $user_i$ and $item_j$ , which you want to be close to $A_{ij}$.

## Why matrix factorization is so efficient?

Matrix factorization typically gives a more `compact representation` than learning the full matrix. The full matrix has entries $O(nm)$, while the embedding matrices $U$, $V$  have $O((n+m)d)$ entries, where the embedding dimension $d$ is typically much smaller than $m$ and $n$, i.e $d \ll m$ and $d \ll n$ . As a result, matrix factorization finds **latent structure** in the data, assuming that observations lie close to a low-dimensional subspace. 

In the preceding example, the values of $n$, $m$, and $d$ are so low that the advantage is negligible. In real-world recommendation systems, however, matrix factorization can be significantly more compact than learning the full matrix.

## Choosing the Objective Function

One intuitive objective function is the squared distance. To do this, minimize the sum of squared errors over all pairs of observed entries:

<center>

$
argmin_{(U, V)} \sum\limits_{i,j \in obs} (A_{ij}- (U_i V_j))^2
$

</center>

In this objective function, you only sum over **observed pairs** $(i, j)$, that is, over non-zero values in the feedback matrix. 

However, only summing over values of one is not a good idea—a matrix of all ones will have a minimal loss and produce a model that can't make effective recommendations and that generalizes poorly.


![image](https://developers.google.com/machine-learning/recommendation/images/UnobservedEntries.svg)

You can solve this quadratic problem through **Singular Value Decomposition** (SVD) of the matrix. However, SVD is not a great solution either, because in real applications, the matrix may be very sparse. For example, think of all the videos on YouTube compared to all the videos a particular user has viewed. The solution

(which corresponds to the model's approximation of the input matrix) will likely be close to zero, leading to poor generalization performance.

In contrast, **Weighted Matrix Factorization** decomposes the objective into the following two sums:

- A sum over **observed entries** i.e. $i,j \in obs$.
- A sum over **unobserved entries** (treated as zeroes) i.e. $ij \notin obs$.

<center>

$argmin_{(U, V)} \sum\limits_{i,j \in obs} (A_{ij}- (U_i V_j))^2 + w_0 \sum\limits_{i,j \notin obs} ((U_i V_j))^2$

</center>

Here $w_0$, is a hyperparameter that weights the two terms so that the objective is not dominated by one or the other. Tuning this hyperparameter is very important.

## Minimizing the Objective Function

Common algorithms to minimize the objective function include:

- :star: **Stochastic gradient descent** (SGD) is a generic method to minimize loss functions.
- :star: **Weighted Alternating Least Squares** (WALS) is specialized to this particular objective.

The objective is quadratic in each of the two matrices $U$ and $V$. (Note, however, that the problem is **not jointly convex**.) 

WALS works by initializing the embeddings randomly, then alternating between:

- Fixing $U$ and solving for $V$
- Fixing $V$ and solving for $U$

Each stage can be solved exactly (via solution of a linear system) and can be distributed. This technique is guaranteed to converge because each step is guaranteed to decrease the loss.

### SGD vs. WALS

SGD and WALS have advantages and disadvantages. Review the information below to see how they compare:

**SGD**

- :heavy_check_mark: Very flexible—can use other loss functions.
- :heavy_check_mark: Can be parallelized.
- :x: Slower—does not converge as quickly.
- :x: Harder to handle the unobserved entries (need to use negative sampling or gravity).

**WALS**

- :x: Reliant on Loss Squares only.
- :heavy_check_mark: Can be parallelized
- :heavy_check_mark: Converges faster than SGD.
- :heavy_check_mark: Easier to handle unobserved entries.

:bookmark_tabs: **References:**

- [CMU Lecture](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf)
- [IMP Google blog - Matrix Factorization from the pov of recsys](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
- [Very good video - Part 1](https://www.youtube.com/watch?v=10qVo3kxAhI)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# What is NMF?

Non-negative matrix factorization (NMF or NNMF), also non-negative matrix approximation is a group of algorithms in multivariate analysis and linear algebra where a matrix $\mathbf{V}$ is factorized into (usually) two matrices $\mathbf{W}$ and $\mathbf{H}$, with the property that all three matrices have `no negative elements`. This non-negativity makes the resulting matrices easier to inspect.

<center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/NMF.png/400px-NMF.png" height="100">
</center>


- Like principal components (SVD), but data and components are
assumed to be non-negative
- Model: $\mathbf{V} \approx \mathbf{W} \mathbf{H}$
- For different application of NMF you can check [this](https://msank00.github.io/blog/2019/07/27/blog_203_ML_NLP_Part_1#what-is-nonnegative-matrix-factorization-nmf-)

<center>
<img src="/assets/images/image_36_mf_1.png" height="380">
</center>

<center>
<img src="/assets/images/image_36_mf_2.png" height="400">
</center>

Figure:1 **Non-negative matrix factorization** (NMF) learns a parts-based representation of faces, whereas vector quantization (VQ) and principal components analysis (PCA) learn holistic representations. 

The three learning methods were applied to a database of $m= 2,429$ facial images, each consisting of $n= 19×19$ pixels, and constituting an $n×m$ matrix $V$. All three find approximate factorizations of the form $X=WH$, but with three different types of constraints on $W$ and $H$, as described more fully in the main text and methods. As shown in the $7×7$ montages, each method has learned a set of $r= 49$ basis images. 

Positive values are illustrated with black pixels :black_circle: and negative values with red pixels :red_circle:. A particular instance of a face :hurtrealbad: , shown at top right, is approximately represented by a linear superposition of basis images. The coefficients of the linear superposition are shown next to each montage, in a $7×7$ grid, and the resulting superpositions are shown on the other side of the equality sign. Unlike VQ and PCA, NMF learns to represent faces with a set of basis images resembling parts of faces.


:bookmark_tabs: **Reference:**

- [Wiki](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
- [Stanford Notes](http://statweb.stanford.edu/~tibs/sta306bfiles/nnmf.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

#  PCA and SVD

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/nbBvuuNVfco" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/nbBvuuNVfco)_


<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/fkf4IBRSeEc" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/fkf4IBRSeEc)_


Follow lecture 6 of [Prof.Mitesh_IITM](https://www.cse.iitm.ac.in/~miteshk/CS7015.html)

+ Eigenvectors can only be found for Square matrix. But, not every square matrix has eigen vectors. 
+ All eigen vectors are perpendicular, i.e orthogonal.
+ orthonormal vectors are orthogonal and they have unit length.
+ If $V$ is an orthonormal matrix then, $V^TV=I$ i.e. $V^T=V^{-1}$

## PCA

+ PCA decomposes a **real, symmetric matrix $A$** into `eigenvectors` and `eigenvalues`. 
+ Every **real, symmetric matrix $A$** can be decomposed into the following expression: $A=VSV^T$.
  + $V$ is an orthogonal matrix. 
  + $S$ is a `diagonal matrix` with all the eigen values.
+ Though, any real symmetric matrix is **guaranteed** to have an **eigen decomposition**, the decomposition may not be unique. 
+ If a matrix is not square, then it's eigen decomposition is not defined.
+ A matrix is `singular` **if and only if**, any of the eigenvalue is zero.
+ Consider A as real, symmetric matrix and $\lambda_i$ are the eigen values.
  + if all $\lambda_i \gt 0$, then $\mathbf{A}$ is called **positive definite** matrix.
  + if all $\lambda_i \ge 0$, then $\mathbf{A}$ is called **positive semidefinite definite (PSD)** matrix.
+ PSD matricies are interesting because the gurantee that `for all x`, $x^TAx \ge 0$. 
+ PSD additionally guarantee that if $x^TAx=0$ then $x=0$. 


:bookmark_tabs: **Reference:**

- [principal-component-analysis](https://medium.com/@SeoJaeDuk/principal-component-analysis-pooling-in-tensorflow-with-interactive-code-pcap-43aa2cee9bb)
- [IMP_link: Deep-Learning-Book-Series-2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.12-Example-Principal-Components-Analysis/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

## SVD 

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/DG7YTlGnCEo" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

>> A is a matrix that can be seen as a linear transformation. This transformation can be decomposed in three sub-transformations: 1. **rotation**, 2. **re-scaling**, 3. **rotation**. These three steps correspond to the three matrices U, D, and V.

<center>

$\mathbf{A} = \mathbf{U D V}^T$

</center>

+ Every matrix can be seen as a linear transformation

>> The transformation associated with diagonal matrices imply only a rescaling of each coordinate without rotation

+ The SVD can be seen as the decomposition of one complex transformation in 3 simpler transformations (a rotation, a scaling and another rotation).
+ SVD is more generic.
+ SVD provides another way for factorizing a matrix, into `singular values` and `singular vectors`.
  + singular values::eigen values = singular vectors::eigen vectors
+ Every real matrix has SVD but same is not true for PCA.
+ If a matrix is not square then PCA not applicable.

- During PCA we write $A=VSV^T$. 
- However, for SVD we write $A=UDV^T$, 
  - where A is `m x n`, U is `m x m`, D is `m x n` and V is `n x n` [**U and V both square !!**]. 
  + U, V orthogonal matricies.
  + D is diagonal matrix and **not necessarily a square**
  + `diag(D)` are the `singular values` of the matrix A
  + Columns of `U` (`V`) are `left (right) singular vectors`.


:bookmark_tabs: **Reference:**

- [Deep-Learning-Book-Series-2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>



----

## Relation between SVD and PCA


>>  The matrices $\mathbf{U}$, $\mathbf{D}$ and $\mathbf{V}$ can be found by transforming $\mathbf{A}$ in a `square matrix` and by computing the `eigenvectors` of this square matrix. The square matrix can be obtain by multiplying the matrix A by its transpose in one way or the other.

+ The `left singular vectors` i.e. $\mathbf{U}$ are the eigen vectors of $\mathbf{A}\mathbf{A}^T$. Similarly, the `right singular vectors` i.e. $\mathbf{V}$ are the eigen vectors of $\mathbf{A}^T\mathbf{A}$. Note that $\mathbf{A}$ might not be a square matrix but $\mathbf{A}\mathbf{A}^T$ or $\mathbf{A}^T\mathbf{A}$ are both square.
  - $\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{D}\mathbf{V}^T$ and $\mathbf{A}\mathbf{A}^T = \mathbf{U}\mathbf{D}\mathbf{U}^T$
 + $\mathbf{D}$  corresponds to the eigenvalues $\mathbf{A}^T\mathbf{A}$ or $\mathbf{A}\mathbf{A}^T$ which are the same.


:bookmark_tabs: **Reference:**

- [source: chapter 2, deep learning book - Goodfellow, p42](http://www.deeplearningbook.org/contents/linear_algebra.html)   


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>



----

## How to prepare data for `PCA`/`SVD` analysis?

Before applying PCA/SVD preprocess with 3 principles:


:radio_button: **Fix count and heavy tail distribution**

- `Sqrt` any features that are `counts`. `log` any feature with a heavy tail.

`Count` data and data with `heavy tails` can mess up the PCA.  

- PCA prefers things that are `homoscedastic`
- `sqrt` and `log` are **variance stabilizing transformations**.  It typically fixes it!


:radio_button: **Localization**

If you make a histogram of a component (or loading) vector and it has really big outliers, that is localization.  It's bad.  It means the vector is noise. 

- **Localization** is noise.  **Regularize** when you normalize.
- To address localization, I would suggest normalizing by **regularized** `row/column sum`

Let $A$ be your matrix, define $rs$ to contain the `row sums`, and $cs$ to contain the `column sums`.  define 

- $D_r = Diag(\frac{1}{\sqrt{(rs + \bar{rs})}})$
- $D_c = Diag(\frac{1}{\sqrt{(cs + \bar{cs})}})$ 

where $\bar{rs}$ is `mean(rs)`

Do SVD on $D_r$   $A$   $D_c$

The use of mean(rs) is what makes it **regularized**.

**Why it works like magic?**

- [NIPS 2018- Understanding Regularized Spectral Clustering viaGraph Conductance](https://papers.nips.cc/paper/8262-understanding-regularized-spectral-clustering-via-graph-conductance.pdf)
  - [Twitter thread](https://twitter.com/karlrohe/status/1011269017582137346?s=20) on above paper summary
  - [Youtube Video](https://www.youtube.com/watch?v=lOCoa3hYR4Y&feature=youtu.be) on the above paper



:radio_button: **The Cheshire cat rule**


```py 
# One day Alice came to a fork in the road and saw a Cheshire cat in a tree.
me: "Which road do I take?" 
cat: "Where do you want to go?" 
me: "I don’t know" 
cat: "Then, it doesn’t matter."
```
In unsupervised learning, we often don't quite know where we are going.

So, is it ok to down-weight, discard, or interact the features? Try it out and see where it takes you!


:bookmark_tabs: **Reference:**

- [twitter thread](https://twitter.com/karlrohe/status/1297741340340629504?s=19)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>



----

## BONUS: Apply SVD on Images:

- [link](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Explain PCA? Tell me the mathematical steps to implement PCA?

+ In PCA, we are interested to find the directions (components) that maximize the variance in our dataset.
+ PCA can be seen as:
  + Learning the projection direction that captures `maximum variance` in data.
  + Learning the projection direction that results in `smallest reconstruction error`
  + Changing the basis in which the data is represented and transforming the features such that new features become `de-correlated` (**Orthogonal Principal Component**)
+ Let’s assume that our goal is to reduce the dimensions of a d-dimensional dataset by projecting it onto a `k`-dimensional subspace (where `k<d`). So, 
how do we know what size we should choose for `k`, and how do we know if we have a feature space that represents our data **well**?
  + We will compute `eigenvectors` (the components) from our data set and 
  collect them in a so-called scatter-matrix (or alternatively calculate 
  them from the **covariance matrix**). 
  + Each of those eigenvectors is associated with an eigenvalue, which tell us about the `length` or `magnitude` of the eigenvectors. 
  + If we observe that all the eigenvalues are of very similar magnitude, this is a good indicator that our data is already in a `good` subspace. 
  + If some of the eigenvalues are much much higher than others, we might be interested in keeping only those eigenvectors with the much larger eigenvalues, since they contain more information about our data distribution. Vice versa, eigenvalues that are close to 0 are less informative and we might consider in dropping those when we construct 
  the new feature subspace.
+ `kth` Eigenvector determines the `kth` direction that maximizes the variance in that direction.
+ the corresponding `kth` Eigenvalue determines the variance along `kth` Eigenvector

:+1: **Tl;DR: Steps**

  + Let $X$ is the original dataset ($n$ x $d$) [$n$: data points, $d$: dimension]
  + Centered the data w.r.t mean and get the centered data: $M$ [dim: $n$ x $d$]
  + Compute covariance matrix: $C=(MM^T)/(n-1)$ [dim: $d$ x $d$]
  + Diagonalize it i.e do `eigen decomposition`: $C = VDV^T$. $V$ is an `orthogonal` matrix so $V^T = V^{-1}$ [[proof](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose)]
    + $V$: [dim: $d$ x $k$]
    + $D$: [dim: $k$ x $k$]
  + Compute `principal component`: $P = V\sqrt{D}$. Or you can also take the first $k$ leading `eigen vectors` from $V$ and the corresponding `eigen values` from $D$ and calculate $P$. [$P$ dim: $d$ x $k$]  
  + Combining all:
  
  <center>

  $C=(MM^T)/(n-1)=VDV^T= = V\sqrt{D}\sqrt{D}V^T =  PP^T$

  </center>
  
  + Apply **principle component matrix** $P$ on the centered data $M$ to get the transformed data projected on the principle component and thus doing `dimensionality reduction`: $M^* = M P$, [$M^*$: new dataset after the PCA, dim: $n$ x $k$]. $k \lt d$, i.e. `dimension reduced` using `PCA`.

:bookmark_tabs: **Resource:**

+ [pca](http://www.math.ucsd.edu/~gptesler/283/slides/pca_15-handout.pdf)
+ [a-one-stop-shop-for-principal-component-analysis](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
+ [rstudio-pubs-static](https://rstudio-pubs-static.s3.amazonaws.com/249839_48d65d85396a465986f2d7df6db73c3d.html)
+ [PPT: Prof. Piyush Rai IIT Kanpur](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec11_slides.pdf)

## What is disadvantage of using PCA?

+ One disadvantage of PCA lies in interpreting the results of dimension reduction analysis. This challenge will become particularly telling when the data needs to be normalized.
+ PCA assumes **approximate normality** of the input space distribution. [link](http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf)
+ For more reading [link](https://www.quora.com/What-are-some-of-the-limitations-of-principal-component-analysis)

## Why PCA needs normalization?

:diamond_shape_with_a_dot_inside: **Mitigate the effects of scale:** For example, if one of the attributes is orders of magnitude higher than others, PCA tends to ascribe the highest amount of variance to this attribute and thus skews the results of the analysis. By normalizing, we can get rid of this effect. **However normalizing results in spreading the influence across many more principal components**. In others words, *more PCs are required to explain the same amount of variance in data*. The interpretation of analysis gets muddied. 


- [Reference](http://www.simafore.com/blog/bid/105347/Feature-selection-with-mutual-information-Part-2-PCA-disadvantages)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Determinant


<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/XkY2DOUCWMU" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/XkY2DOUCWMU)_



<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/Ip3X9LOh2dk" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/Ip3X9LOh2dk)_


>> The determinant of a matrix A is a number corresponding to the **multiplicative change** you get when you transform your space with this matrix.

+ A **negative determinant** means that there is a **change in orientation** (and not just a rescaling and/or a rotation).

:bookmark_tabs: **Reference:**

- [(Important_link)](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.11-The-determinant/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----
# Recommendation System:

## Content based recommendation:

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/2uxXPzm-7FY" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/2uxXPzm-7FY)_

### Using Feature vector and regression based analysis

Here the assumption is that for each of the `item` you have the corresponding features available. 

:tada: **Story:** Below are a table, of movie-user rating. 5 users and 4 movies. And the ratings are also provided (1-5) for some of them. We want to predict the rating for all the `?` unknown. It means that user $u_j$ has not seen that movie $m_i$ and if we can predict the rating for cell (i,j) then it will help us to decide whether or not to recommend the movie to the user $u_j$. For example, cell (3,4) is unknown. If by some algorithm, we can predict the rating for that cell, say the predicted rating is 4, it means if the user **would have seen this movie, then he would have rated the movie 4**. So we must definitely recommend this movie to the user. 


|    | u1 | u2 | u3 | u4 | u5 |
|:--:|:--:|:--:|:--:|----|:--:|
| m1 |  4 |  5 |  ? | 1  |  2 |
| m2 |  ? |  4 |  5 | 0  |  ? |
| m3 |  0 |  ? |  1 | ?  |  5 |
| m4 | 1  | 0  | 2  | 5  | ?  |

For content based movie recommendation it's assumed that the features available for the content. For example if we see closely, then we can see the following patterns in the table. The bold ratings have segmented the table into 4 sub parts based on rating clustering.

|    | u1 | u2 | u3 | u4 | u5 |
|:--:|:--:|:--:|:--:|----|:--:|
| m1 |  **4** |  **5** |  **?** | 1  |  2 |
| m2 |  **?** |  **4** |  **5** | 0  |  ? |
| m3 |  0 |  ? |  1 | **?**  |  **5** |
| m4 | 1  | 0  | 2  | **5**  | **?**  |


as if movie $m_1$, $m_2$ belong to type 1 (say romance) and $m_3$, and $m_4$ belong to type 2 (say action) and there is a clear discrimination is the rating as well.

Now for content based recommendation this types are available and then the datasets actually looks as follows

|    | u1 | u2 | u3 | u4 | u5 | T1  | T2  |
|:--:|:--:|:--:|:--:|----|:--:|-----|-----|
| m1 |  4 |  5 |  ? | 1  |  2 | 0.9 | 0.1 |
| m2 |  ? |  4 |  5 | 0  |  ? | 0.8 | 0.2 |
| m3 |  0 |  ? |  1 | ?  |  5 | 0.2 | 0.8 |
| m4 | 1  | 0  | 2  | 5  | ?  | 0.1 | 0.9 |

where $T_1$ and $T_2$ columns are already known. Then for each of the user we can learn a regression problem with the known rating as the target vector $u_j$ and $A = [T_1,T_2]$
is the feature matrix and we need to learn the $\theta_j$ for user $j$ such that $A \theta_j = u_j$. Create the loss function and solve the optimization problem.

:bookmark_tabs: **Reference:**

- [(Content Based Recom -A.Ng)](https://www.youtube.com/watch?v=c0ZPDKbYzx0&list=PLnnr1O8OWc6ZYcnoNWQignIiP5RRtu3aS&index=2),
- [(MMD - Stanford )](https://www.youtube.com/watch?v=2uxXPzm-7FY&index=42&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV)



## Colaborative Filtering

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/h9gpufJFF-0" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/h9gpufJFF-0)_



<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/v_mONWiFv0k" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/v_mONWiFv0k)_



:tada: **Story:** Unlike the `content based recommendation`, where the feature columns ($T_1$, $T_2$) were already given, here the features are not present. Rather they are being learnt by the algorithm. Here we assume $\theta_j$ is given i.e we know the users liking for $T_1$ and $T_2$ movies and almost similarly we formulate the regression problem but now we try to estimate the feature columns $T_k$. Then using the learnt feature we estimate the unknown ratings and then recommend those movies. Here knowing $\theta_j$
 means that each user has given information regarding his/her preferences based on subset of movies and thus all the users are helping in colaborative way to learn the features.

:large_orange_diamond: **Naive Algorithm:**

1. Given $\theta_j$, learn features $T_k$. [loss function $J(T_k)$]
2. Given $T_k$, learn $\theta_j$. [loss function $J(\theta_j)$] 

and start with a random initialization of $\theta$ and then move back and forth between step 1 and 2.

:large_orange_diamond: **Better Algorithm:**

- combine step 1 and 2 into a single loss function $J(\theta_j,T_k)$ and solve that.


### User User Colaborative Filtering

- Fix a similarity function (Jaccard similarity, cosine similarity) for two vectors. and then pick any two users profile (their rating for different movies) and calculate the similarity function which helps you to cluster the users.
  - `Jaccard similarity` doesn't consider the rating
  - `Cosine similarity` consider the unknown entries as 0, which causes problem as rating range is (0-5).
  - `Centered` cosine similarity (pearson correlation): subtract $\mu_{user}^{rating}$ from $rating_{user}$. 
    - Missing ratings are treated as average 

### Item Item Colaborative Filtering

Almost similar to User User CF.

:+1: **Pros:**

- Works for any kind of item
- No need data for other user
- It's personalized recommendation as for each user a regression problem is learnt.
- No first-rater problem, i.e. we can recommend an item to the user, as soon as it's available in the market.

:-1: **Cons:**

- Finding the correct feature is hard 
- Cold start problem for new user 
- Sparsity
- Popularity bias

:bookmark_tabs: **Reference:**

- [(Collaborative Filtering-A.Ng)](https://www.youtube.com/watch?v=-Fptv3NZtmE&index=4&list=PLnnr1O8OWc6ZYcnoNWQignIiP5RRtu3aS)
- [(MMD - Stanford )](https://www.youtube.com/watch?v=2uxXPzm-7FY&index=42&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV)

----

## Colaborative Filtering - Low Rank Matrix Factorization (SVD)

### Evaluation of Recommending System

RMSE: Root Mean Square Error. However it doesn't distinguish between high rating and low rating. 

- Alternative: Precision $@k$ i.e. get the precision for top k items.

## Deep Learning based Recommendation System

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/ZkBQ6YA9E40" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/ZkBQ6YA9E40)_



<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/MVB1cbe923A" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/MVB1cbe923A)_


- For more on Recommendation System, click [here](https://msank00.github.io/blog/2019/11/18/blog_104_RecSys). :rocket:

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Solve Linear Programming

>> :sparkles: min $c^Tx$ subject to: $Ax = b$, $x ≥ 0$.

The linear programming problem is usually solved through the use of one of two
algorithms: either **simplex**, or an algorithm in the family of **interior point methods**.

In this article two representative members of the family of interior point methods are
introduced and studied. We discuss the design of these interior point methods on a high
level, and compare them to both the simplex algorithm and the original algorithms in
nonlinear constrained optimization which led to their genesis.
:book: [[survey paper]](https://www.cs.toronto.edu/~robere/paper/interiorpoint.pdf)



## Simplex Method

> Let $A, b, c$ be an instance of the LPP, defining a convex polytope in $R^n$. Then there exists an optimal solution to this program at one of the vertices of the **polytope** :sparkles: .

The simplex algorithm works roughly as follows. We begin with a feasible point at one
of the vertices of the polytope. Then we “walk” along the edges of the polytope from vertex
to vertex, in such a way that the value of the objective function monotonically decreases at
each step. When we reach a point in which the objective value can decrease no more, we are
finished. 

:bookmark_tabs: **Reference:**

- [Youtube](https://www.youtube.com/watch?v=vVzjXpwW2xI)
- [AV](https://www.analyticsvidhya.com/blog/2017/02/lintroductory-guide-on-linear-programming-explained-in-simple-english/)
- [Paper](https://www.cs.toronto.edu/~robere/paper/interiorpoint.pdf)

## Interior Point Method

Our above question about the complexity of the LPP was answered by Khachiyan in 1979. He demonstrated a worst-case polynomial time algorithm for linear programming dubbed the ellipsoid method [Kha79] in which the algorithm moved across the **interior of the feasible region, and not along the boundary like simplex**. Unfortunately the worst case running time for the ellipsoid method was high: $O(n^6L^2)$, where n is the number of variables in the problem
and L is the number of the bits in the input. Moreover, this method tended to approach the
worst-case complexity on nearly all inputs, and so the simplex algorithm remained dominant
in practice. This algorithm was only partially satisfying to researchers: was there a worst-case
polynomial time algorithm for linear programming which had a performance that rivalled
the performance of simplex on day-to-day problems?
This question was answered by Karmarkar in 1984. He produced a polynomial-time
algorithm — soon called the projective algorithm — for linear programming that ran in
much better time: $O(n^{3.5}L^2)$ [Kar84]. 

:bookmark_tabs: **Reference:**

- [(link1)](https://www.inf.ethz.ch/personal/fukudak/lect/opt2011/aopt11note4.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

## First Order Method -  The KKT Conditions and Duality Theory

![image](/assets/images/image_25_svm_kkt_1.png)

- [Prof. SB IIT KGP, course](http://cse.iitkgp.ac.in/~sourangshu/coursefiles/ML15A/svm.pdf)
- [Imp Lecture Notes](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf)



![image](/assets/images/image_25_svm_lagrange_1.png)
![image](/assets/images/image_25_svm_lagrange_2.png)


:star: **KKT Conditions:**

- Stationarity $\nabla_x L(x,\lambda,\mu)=0$
- Primal feasibility, $g(x) \leq 0$ (for minimization problem)
- Dual feasibility, $\lambda \geq 0, \mu \geq 0$
- Complementary slackness, $\lambda g(x) = 0$ and $\mu h(x)=0$

:bookmark_tabs: **Resource**

- [Prof. Piyush Rai, Lecture 10](https://www.cse.iitk.ac.in/users/piyush/courses/ml_autumn18/material/771_A18_lec10_print.pdf)
- [notes](http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap4.pdf)
- [sb slides](http://cse.iitkgp.ac.in/~sourangshu/coursefiles/ML15A/svm.pdf)
- [iitK_notes](https://www.cse.iitk.ac.in/users/rmittal/prev_course/s14/notes/lec11.pdf)
- [Khan Academy - Constrained Optimization](https://www.youtube.com/watch?v=vwUV2IDLP8Q&list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7&index=92)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# What is LBFGS algorithm?

- In numerical optimization, the `Broyden–Fletcher–Goldfarb–Shanno` (BFGS) algorithm is an `iterative method` for solving `unconstrained nonlinear optimization` problems.
- The BFGS method belongs to quasi-Newton methods, a class of hill-climbing optimization techniques that seek a stationary point of a (preferably twice continuously differentiable) function. For such problems, a necessary condition for optimality is that the gradient be zero. Newton's method and the BFGS methods are not guaranteed to converge unless the function has a quadratic Taylor expansion near an optimum. However, BFGS can have acceptable performance even for non-smooth optimization instances.

:bookmark_tabs: **Resource:**

- [Wikipedia](https://www.google.com/search?q=BFGS+algorithm&ie=utf-8&oe=utf-8&client=firefox-b-e&safe=strict)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


# Second-Order Methods:  Newton’s Method


- GD and variants only use first-order information (the gradient)
- :rocket: **Second-order** information often tells us a lot more about the function’s `shape`, `curvature`, etc.
  - **Newton’s method** is one such method that uses second-order information.
- At each point, approximate the function by its quadratic approx. and minimize it
- Doesn’t rely on gradient to choose $w_{(t+1)}$Instead, each step directly jumps to the minima of quadratic approximation.
- No learning rate required  :+1:
- Very fast if $f(w)$ is convex.  But expensive due to Hessian computation/inversion.
- Many ways to approximate the Hessian (e.g., using previous gradients); also look at `L-BFGS`

## Second order Derivative

- In calculus, the second derivative, or the second order derivative, of a function $f$ is the derivative of the derivative of $f$. Roughly speaking, the second derivative measures how the `rate of change of a quantity` is itself `changing`; 
  - For example, the second derivative of the position of a vehicle with respect to time is the instantaneous acceleration of the vehicle, or the rate at which the velocity of the vehicle is changing with respect to time.

:atom_symbol: On the `graph of a function`, the second derivative corresponds to the `curvature` or `concavity of the graph`. 
- The graph of a function with a `positive second derivative` is `upwardly concave`, 
- The graph of a function with a `negative second derivative` curves in the opposite way.

![animation](https://upload.wikimedia.org/wikipedia/commons/7/78/Animated_illustration_of_inflection_point.gif)

> A plot of $f( x ) = sin ⁡ ( 2 x )$  from $-\pi/4$ to $5\pi/4$ . The tangent line is blue where the curve is `concave up`, green where the curve is `concave down`, and red at the inflection points (0, $\pi/2$, and $\pi$) :sparkles: .

:bookmark_tabs: **Reference:**

- [Prof Piyush Rai, IIT Kanpur](https://www.cse.iitk.ac.in/users/piyush/courses/ml_autumn18/material/771_A18_lec9_print.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Gradient, Jacobian, Hessian, Laplacian and all that

:atom_symbol: **Gradient:** To generalize the notion of derivative to the multivariate functions we use the gradient operator.


<center>

$\nabla f = [ \frac{\delta f(x_1, x_2)}{\delta x_1} , \frac{\delta f(x_1, x_2)}{\delta x_2} ]$

</center>


Gradient vector gives us the **magnitude** and `direction of maximum change` of a multivariate function. 

:atom_symbol: **Jacobian:** The Jacobian operator is a generalization of the derivative operator to the vector-valued functions. As we have seen earlier, a vector-valued function is a mapping from $f:ℝ^n \rightarrow ℝ^m$ hence, now instead of having a scalar value of the function $f$, we will have a mapping $[x_1,x_2, \dots ,x_n] \rightarrow [f_1,f_2, \rightarrow,f_n]$. Thus, we now need the rate of change of each component of $f_i$ with respect to each component of the input variable $x_j$, this is exactly what is captured by a matrix called Jacobian matrix $J$:

<center>

$
J = \begin{vmatrix}
\frac{\delta f_1}{\delta x_1} , \frac{\delta f_1}{\delta x_2}\\
\frac{\delta f_2}{\delta x_1} , \frac{\delta f_2}{\delta x_2}\\
\end{vmatrix}
$


</center>


:atom_symbol: **Hessian:** The gradient is the first order derivative of a multivariate function. To find the second order derivative of a multivariate function, we define a matrix called a Hessian matrix. It describes the `local` **curvature** of a function of many variables. 

Suppose $f : ℝ^n \rightarrow ℝ$ is a function taking as input a vector $x ∈ ℝ^n$ and outputting a scalar $f(x) ∈ ℝ$. If all second partial derivatives of $f$ exist and are continuous over the domain of the function, then the Hessian matrix $H$ of f is a square $n \times n$ matrix, usually defined and arranged as follows

<p align="center">

$
H = \begin{vmatrix}
\frac{\delta^2 f}{\delta x_1^2} , \frac{\delta^2 f}{\delta x_1 x_2}\\
\frac{\delta^2 f}{\delta x_2 x_1} , \frac{\delta^2 f}{\delta^2 x_2}\\
\end{vmatrix}
$

</p>

The Hessian matrix of a function f is the Jacobian matrix of the gradient of the function f ; that is: $H(f(x)) = J(\nabla f(x))$. 

:atom_symbol: **Laplacian:** The trace of the Hessian matrix is known as the Laplacian operator denoted by $\nabla^2$

<center>

$
\nabla ^2 f = trace(H) = \Sigma_i \frac{\delta^2 f}{\delta x_i^2}
$

</center>

**Reference:**

- [link](https://najeebkhan.github.io/blog/VecCal.html)

-----

# Why does L1 regularization induce sparse models?

- [Tweet](https://twitter.com/itayevron/status/1328421322821693441?s=20) :fire:


<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/76B5cMEZA4Y" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/76B5cMEZA4Y)_


<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/B5Rd2ctb_kg" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/B5Rd2ctb_kg)_




----


# Clustering Algorithm

- When data set is very large, can't fit into memory, then K means algorithm is not a good algorithm. Rather use Bradley-Fayyad-Reina (BFR) algorithm.

## BFR (Bradley, Fayyad and Reina)

The BFR algorithm, named after its inventors Bradley, Fayyad and Reina, is a **variant of k-means algorithm** that is designed to cluster data in a high-dimensional Euclidean space. It makes a very `strong assumption` about the shape of clusters: 
- They **must be normally distributed about a centroid** :rocket: .
- The mean and standard deviation for a cluster may differ for different dimensions, but the dimensions must be independent.

:dart: **Strong Assumption**:

- Each cluster is `normally distributed` around a centroid in Euclidean space
- Normal distribution assumption implies that clusters looks like `axis-aligned ellipses`, i.e. standard deviation is maximum along one axis and minimum w.r.t other.
 
:bookmark_tabs: **Reference:**

- [MMD Youtube](https://www.youtube.com/watch?v=NP1Zk8MY08k)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to implement regularization in decision tree, random forest?

In **Narrow sense**, `Regularization` (commonly defined as adding Ridge or Lasso Penalty) is difficult to implement for Trees. 
- Tree is a heuristic algorithm.

In **broader sense**, Regularization (as any means to prevent overfit) for Trees is done by:

- Limit max. depth of trees
- Ensembles / bag more than just 1 tree
- Set stricter stopping criterion on when to split a node further (e.g. min gain, number of samples etc.)

:bookmark_tabs: **Resource:**

- [Quora](https://www.quora.com/How-is-regularization-performed-on-simple-decision-trees)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Why random column selection helps random forest?

Random forest algorithm comes under the `Bootstrap Algorithm` a.k.a `Bagging` category whose primary objective is to `reduce the variance` of an estimate by averaging many estimates.

- :star: **Sample datasets without replacement**
  - In bootstarp algorithm `M` decision trees are trained on `M` datasets which are sampled `without replacement` from the original datasets. Hence each of the sampled data will have duplicate data as well as some missing data, which are present in the original data, as all the sampled datasets have same length. Thus, these bootstrap samples will have diversity among themselves, resulting the model to be diverse as well. 

Along with the above methods `Random Forest` also does the following:

- :star: **Random subset of feature selection** for building the tree
  - This is known as subspace sampling
  - Increases the diversity in the ensemble method even more
  - Tree training time reduces.
  - :fire: **IMPORTANT:** Also simply running the same model on different sampled data will produce highly correlated predictors, which limits the amount of variance reduction that is possible. So RF tries to **decorrelate** the base learners by learning trees based on a randomly chosen subset of input variables, as well as, randomly chosen subset of data cases.

:bookmark_tabs: **Reference:**

- source: 1. Kevin Murphy, 2. Peter Flach
- [ref](https://www.quora.com/How-does-bagging-avoid-overfitting-in-Random-Forest-classification)

## Overfitting in Random Forest

- To avoid over-fitting in random forest, the main thing you need to do is optimize a tuning parameter that governs the number of features that are randomly chosen to grow each tree from the bootstrapped data. Typically, you do this via k-fold cross-validation, where k∈{5,10}, and choose the tuning parameter that minimizes test sample prediction error. In addition, growing a larger forest will improve predictive accuracy, although there are usually diminishing returns once you get up to several hundreds of trees.

- For decision trees there are two ways of handling overfitting: (a) don't grow the trees to their entirety (b) prune. The same applies to a forest of trees - don't grow them too much and prune.


:bookmark_tabs: **Reference:**

- [SO:random-forest-tuning-tree-depth-and-number-of-trees](https://stackoverflow.com/questions/34997134/random-forest-tuning-tree-depth-and-number-of-trees/35012011#35012011)
- [statsexchange:random-forest-how-to-handle-overfitting](https://stats.stackexchange.com/questions/111968/random-forest-how-to-handle-overfitting)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to avoid Overfitting?

Overfitting means High Variance

Steps to void overfitting:

- Cross Validation
- Train with more data
- Remove feature
- Early stopping
- Regularizations
- Ensembling

[Ref 1](https://elitedatascience.com/overfitting-in-machine-learning)

# How to avoid Underfitting?

- Add more features
- Add more data
- Decrease the amount of regularizations

[ref 1](https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

------

# Gaussian Processing

![image](/assets/images/image_11_GP_1.png)
![image](/assets/images/image_11_GP_2.png)
![image](/assets/images/image_11_GP_3.png)
![image](/assets/images/image_11_GP_4.png)
![image](/assets/images/image_11_GP_5.png)


- GP: Collection of random variables, any finite number of which are Gaussian Distributed
- Easy to use: Predictions correspond to models with infinite parameters
- Problem: $N^3$ complexity
   

:bookmark_tabs: **Reference:**

- [Slide: Prof. Richard Turner](http://gpss.cc/gpss13/assets/Sheffield-GPSS2013-Turner.pdf)
- [Lecture Video: Prof. Richard Turner](https://www.youtube.com/watch?v=92-98SYOdlY)
- [Gaussian Process](http://www.gaussianprocess.org/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# K-Means Clustering (Unsupervised Learning)

![image](/assets/images/image_13_KMeans_1.png)
![image](/assets/images/image_13_KMeans_2.png)
![image](/assets/images/image_13_KMeans_3.png)
![image](/assets/images/image_13_KMeans_4.png)
![image](/assets/images/image_13_KMeans_5.png)
![image](/assets/images/image_13_KMeans_6.png)


:bookmark_tabs: **Resource:**

- [Prof. Piyush Rai, Lecture 10](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec10_slides.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# PCA and Kernel PCA


- [Prof. Piyush Rai, Lecture 11](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec11_slides.pdf)

----

# Generative Model (Unsupervised Learning)


![image](/assets/images/image_12_GenModel_1.png)
![image](/assets/images/image_12_GenModel_2.png)
![image](/assets/images/image_12_GenModel_3.png)
![image](/assets/images/image_12_GenModel_4.png)
![image](/assets/images/image_12_GenModel_5.png)
![image](/assets/images/image_12_GenModel_6.png)


:bookmark_tabs: **Resource:**

- [Prof. Piyush Rai, Lecture 15](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec15_slides.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---- 

# Ensemble Method (Bagging and Boosting)

![image](/assets/images/image_15_Ensemble_1.png)
![image](/assets/images/image_15_Ensemble_2.png)
![image](/assets/images/image_15_Ensemble_3.png)
![image](/assets/images/image_15_Ensemble_4.png)
![image](/assets/images/image_15_Ensemble_5.png)
![image](/assets/images/image_15_Ensemble_6.png)
![image](/assets/images/image_15_Ensemble_7.png)


:bookmark_tabs: **Reference:**

- [Prof. Piyush Rai, Lecture 21](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec21_slides.pdf)
- [Youtube Abhishek Thakur](https://www.youtube.com/watch?v=TuIgtitqJho) :rocket:

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Online Learning

![image](/assets/images/image_14_OnlineLearning_1.png)
![image](/assets/images/image_14_OnlineLearning_2.png)
![image](/assets/images/image_14_OnlineLearning_3.png)
![image](/assets/images/image_14_OnlineLearning_4.png)

:bookmark_tabs: **Resource:**

- [Prof. Piyush Rai, Lecture 26](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec26_slides.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Model/ Feature Selection and Model Debugging

![image](/assets/images/image_16_ModelSelection_1.png)

**Different strategies:**

- Hold-out/ Validation data
  - Problem: Wastes training time
- K-Fold Cross Validation (CV)
- Leave One Out (LOO) CV
  - Problem: Expensive for large $N$.
  - But very efficient when used for selecting the number of neighbors to consider in nearest neighbor methods. (reason:  NN methods require no training)
- Random Sub-sampling based Cross-Validation
- Bootstrap Method
  
![image](/assets/images/image_16_ModelSelection_2.png)

- Information Criteria based methods
  
![image](/assets/images/image_16_ModelSelection_3.png)

## Debugging Learning Algorithm

![image](/assets/images/image_16_ModelSelection_4.png)
![image](/assets/images/image_16_ModelSelection_5.png)
![image](/assets/images/image_16_ModelSelection_6.png)
![image](/assets/images/image_16_ModelSelection_7.png)


**Resource:**

- [Prof. Piyush Rai, Lecture 19](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec19_slides.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Bayesian Machine Learning

Coursera - Bayesian Methods for Machine Learning


# Finding similar sets from millions or billions data

Techniques:
1. Shingling: converts documents, emails etc to set.
2. Min-hashing: Convert large sets to short signature, while preserving similarity
3. Locality-Sensitive-Hashing: Focus on pair of signatures likely to be similar.

:bookmark_tabs: **Reference:**

Lecture 12, 13, 14 of below playlist

- [Mining Massive Datasets - Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV)

----

# :star: Top Pick for Interview - Depth and Breadh

- [Entropy](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#what-is-the-formula-for-entropy-criteria)
- [KL divergence](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#what-is-kl-divergence)
  - [other divergences](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#divergence)
- [Kolmogorov complexity](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#kolmogorov-complexity)
- [Jacobian and Hessian](https://msank00.github.io/blog/2019/07/04/blog_100_ML_Concepts_Part_1#gradient-jacobian-hessian-laplacian-and-all-that)
- Linear independence
- Determinant
- [Eigenvalues and Eigenvectors](https://msank00.github.io/blog/2019/07/04/blog_100_ML_Concepts_Part_1#eigen-decomposition)
- [PCA and SVD](https://msank00.github.io/blog/2019/07/04/blog_100_ML_Concepts_Part_1#pca-and-svd)
- The norm of a vector
- Independent random variables
- Expectation and variance
- [Central limit theorem](https://msank00.github.io/blog/2019/07/20/blog_001_Statistical_Analysis_Part_1#central-limit-theorem)
- [Law of Large Number](https://msank00.github.io/blog/2019/07/20/blog_001_Statistical_Analysis_Part_1#law-of-large-number) 
- [Gradient descent and SGD](https://msank00.github.io/blog/2019/07/16/blog_102_DL_Concepts_Part_1#gradient-decent)
  - [Other optimization methods](https://msank00.github.io/blog/2019/07/16/blog_102_DL_Concepts_Part_1#momentum)
- The dimension of gradient and hessian for a neural net with 1k params 
- [What is SVM](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#svm-summary)
  - [Linear](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#algorithm) vs [non-linear SVM](https://msank00.github.io/blog/2019/07/05/blog_101_ML_Concepts_Part_2#geometric-analysis-of-lagrangian-kkt-dual)
- Quadratic optimization
- What to do when a neural net overfits
- What is autoencoder
- How to train an RNN
- How decision trees work
- Random forest and GBM
- How to use random forest on data with 30k features
- Favorite ML algorithm - tell about it in details



----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>



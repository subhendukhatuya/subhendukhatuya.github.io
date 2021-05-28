---
layout: post
title:  "Machine Learning System Design (Part - 1)"
date:   2019-11-30 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
----

# Introduction 

To learn how to design machine learning systems

> I find it really helpful to read case studies to see how great teams deal with different deployment requirements and constraints.

- [Tweet: Chip Huyen](https://twitter.com/chipro/status/1188653180902445056)


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Model Deployment

## Model compression & optimization

> :bulb: Google: Latency 100 -> 400 ms reduces searches 0.2% - 0.6% (2009)

> :bulb: booking.com:  30% increase in latency costs 0.5% conversion rate (2019)

No matter how great your ML models are, if they take just milliseconds too long,
users will click on something else.

**Fast inference**

- Make models **faster**

![image](/assets/images/image_44_mlsd_3.png)

- Make models **smaller**

![image](/assets/images/image_44_mlsd_2.png)

[model_compression](https://awesomeopensource.com/projects/model-compression) :rocket:, [pruning](https://arxiv.org/abs/1506.02626) :book:, [factorization](https://arxiv.org/abs/1704.04861) :book:,



- Make hardware more `powerful`





## Simple deployment

![image](/assets/images/image_44_mlsd_1.png)

## Containerized deployment

- **Reduced complexity:** each developer works on a smaller codebase
- **Faster development cycle:** easier review process
- **Flexible stack:** different microservices can use different technology stacks. [Why to use microservice?](https://developer.ibm.com/depmodels/microservices/articles/why-should-we-use-microservices-and-containers/)
- [Digging into Docker](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612), Kubernetes, [link](https://www.docker.com/blog/top-questions-docker-kubernetes-competitors-or-together/)

<center>

<img src="https://i1.wp.com/www.docker.com/blog/wp-content/uploads/2019/10/Docker-Kubernetes-together.png?resize=1110%2C624&ssl=1" width="500">

</center>


<center>

<img src="https://i2.wp.com/www.docker.com/blog/wp-content/uploads/2019/10/Kubernetes-with-Docker-Enterprise.png?resize=1110%2C624&ssl=1" width="500">

</center>


## Test in production

**Canary Testing**

- New model alongside existing system
- Some traffic is routed to new model
- Slowly increase the traffic to new model
  - E.g. roll out to Vietnam first, then Asia, then rest of the world


<center>

<img src="https://cloud.google.com/solutions/images/application-deployment-and-testing-strategies-canary-deployment.svg" width="650">

</center>

**A/B testing**

- New model alongside existing system
- A percentage of traffic is routed to new model based on routing rules
- Control target audience & monitor any statistically significant differences in user behavior
- Can have more than 2 versions

<center>

<img src="https://cloud.google.com/solutions/images/application-deployment-and-testing-strategies-A-B-testing.svg" width="300">

</center>


**Shadow test pattern**


- New model in parallel with existing system
- New model’s predictions are logged, but not show to users
- Switch to new model when results are satisfactory


<center>

<img src="https://cloud.google.com/solutions/images/application-deployment-and-testing-strategies-shadow-testing.svg" width="300">

</center>


## Different deployment strategies

- [link](https://cloud.google.com/solutions/application-deployment-and-testing-strategies#deployment_strategies)


**Reference:**

- [CS 329S: Machine Learning Systems Design Winter 2021 - Chip Huyen](https://stanford-cs329s.github.io/syllabus.html)
- [Application deployment and testing strategies - Google](https://cloud.google.com/solutions/application-deployment-and-testing-strategies) :fire: :fire:


----

# How to test your machine learning system?

A typical software testing suite will include:

- **unit tests** which operate on atomic pieces of the codebase and can be run quickly during development,
- **regression tests** replicate bugs that we've previously encountered and fixed,
- **integration tests** which are typically longer-running tests that observe higher-level behaviors that leverage multiple components in the codebase,


Let's contrast this with a typical workflow for developing machine learning systems. After training a new model, we'll typically produce an **evaluation report** including:

- Performance of an established metric on a validation dataset,
- Plots such as precision-recall curves,
- Operational statistics such as inference speed,
- Examples where the model was most confidently incorrect,

and follow conventions such as:

- Save all of the hyper-parameters used to train the model,
- Only promote models which offer an improvement over the existing model (or baseline) when evaluated on the same dataset. 

![image](https://www.jeremyjordan.me/content/images/size/w1000/2020/08/Group-3-1.png)


>> it feels like that testing for machine learning systems is in such early days that this question of test coverage isn't really being asked by many people. :star:


## Difference between model testing and model evaluation

For machine learning systems, we should be running model evaluation and model tests in parallel.

- **Model evaluation** covers metrics and plots which summarize performance on a validation or test dataset.
- **Model testing** involves explicit checks for behaviors that we expect our model to follow.

**NOTE:** Do [error analysis](https://www.coursera.org/lecture/machine-learning-projects/carrying-out-error-analysis-GwViP)

## How do you write model tests?

There's two general classes of model tests that we'll want to write.

- **Pre-train tests** allow us to identify some bugs early on and short-circuit a training job.
- **Post-train tests** use the trained model artifact to inspect behaviors for a variety of important scenarios that we define.

_*please read the actual blog thoroughly_

- **Invariance Tests:** check for consistency in the model predictions
- **Directional Expectation Tests:**  define a set of perturbations to the input which should have a predictable effect on the model output. 
  - Increasing the number of bathrooms (holding all other features constant) should not cause a drop in price.
  - Lowering the square footage of the house (holding all other features constant) should not cause an increase in price.
- **Minimum Functionality Tests** (aka data unit tests): 

## Model development pipeline

![image](https://www.jeremyjordan.me/content/images/size/w1000/2020/08/Group-7.png)

**Reference:**

- [Effective testing for machine learning systems.](https://www.jeremyjordan.me/testing-ml/) :fire: :fire:


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>



----

# 12 Factors of reproducible Machine Learning in production

<center>

<img src="https://zenml.io/assets/illustrations/architecture/vertical-large.svg" width="500">

</center>

1. You need to version your code, and you need to version your data.

> :bulb: Serverless functions can provide an easy-access way to achieve a middle ground between the workflow of Data Scientists and production-ready requirements.

2. Make your feature dependencies explicit in your code

> :bulb: Explicitly defined feature dependencies allow for transparent failure as early as possible. Well-designed systems will accommodate feature dependencies both in continuous training as well as at serving time.

3. Write readable code and separate code from configuration.

> :dart: Code for both preprocessing and models should follow PEP8. It should consist of meaningful object names and contain helpful comments. Following `PEP8` will improve code legibility, reduce complexity and speed up debugging. Programming paradigms such as `SOLID` provide thought frameworks to make code more maintainable, understandable and flexible for future use cases.

4. :fire: Reproducibility of trainings - Use **pipelines** and **automation**.

> By using `pipelines` to train models entire teams gain both access and transparency over conducted experiments and training runs. Bundled with a `reusable codebase` and a `separation from configuration`, everyone can successfully relaunch any training at any point in time.

5. Test your code, test your models.
6. :fire: Drift / Continuous training - If you data can change run a **continuous training pipeline**

> Data monitoring for production systems. Establish automated reporting mechanisms to alert teams of changing data, even beyond explicitly defined feature dependencies
> Continuous training on newly incoming data. Well-automated pipelines can be rerun on newly recorded data and offer comparability to historic training results to show performance degradation

7. Track results via automation - **Weights and Bias**

8. Experimentation vs Production models - Notebooks are not production-ready, so experiment in pipelines early on

> :bulb: Gained understanding (through Notebook) will need more molding and fitting into production-ready training pipelines. All understandings unrelated to domain-specific knowledge can however be automated.

**The earlier you experiment in pipelines, the earlier you can collaborate on intermediate results and the earlier you’ll receive production-ready models**

9. Training-Serving-Skew - Correctly embed preprocessing to serving, and make sure you understand up- and downstream of your data


10. Build your pipelines so you can easily compare training results across pipelines.

> :bulb: the technical possibility to compare model trainings needs to be built into training architecture as a first-class citizen early on

11. Again: you build it, you run it. Monitoring models in production is a part of data science in production

> Plenty of negative degradations can occur in the lifecycle of a model: Data can drift, models can become bottlenecks for overall performance and bias is a real issue. Data Scientists and teams are responsible for monitoring the models they produce. **At its minimum**, a model needs to be monitored for input data, inference times, resource usage (read: CPU, RAM) and output data 

12. Every training pipeline needs to produce a deployable artefact, not “just” a model

**Reference:**

- [12 Factors of reproducible Machine Learning in production](https://blog.maiot.io/12-factors-of-ml-in-production/)
- [ZenML](https://zenml.io/#features) :fire:
  - [Why ZenML](https://zenml.io/why-ZenML/)
- [Seldon Core](https://github.com/SeldonIO/seldon-core)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# MLflow and PyTorch — Where Cutting Edge AI meets MLOps

Full workflow for MLops for building PyTorch models and deploying on the TorchServe plugin using the full MLflow MLops lifecycle management

<center>

<img src="https://miro.medium.com/max/700/0*R06hbsJNmes9GfGS" width="500">

</center>

- [Blog](https://medium.com/pytorch/mlflow-and-pytorch-where-cutting-edge-ai-meets-mlops-1985cf8aa789)
- [ BERT News Classification example](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/BertNewsClassification)

----

# Practical Deep Learning Systems - Full Stack Deep Learning

**Play the video:** :arrow_heading_down: 

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/5ygO8FxNB8c" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

>> The average elapsed time between key algorithm proposal and corresponding advances was about 18 years, whereas the average elapsed time between key dataset availabilities and corresponding advances was less than 3 years, i.e. about 6 times faster. 

- More Data or Better data?
- Simple model >> Complex Model
- ... but sometime you do need complex model
- We should care about feature engineering
- Deep Learning: <del>Feature</del> **Architecture** Engineering
- Supervised/Unsupervised => Self-Supervised learning
- Everything is ensemble
- There is bias in your data
- Curse of presentation bias
- Bias and Fairness
- Right Evaluation: Offline and Online Experimentation
- ML beyond DL


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Serve model using Torchserve

- [Getting started with Torchserve](https://cceyda.github.io/blog/torchserve/streamlit/dashboard/2020/10/15/torchserve.html)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
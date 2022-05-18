---
layout: default
title: Projects
#permalink: /publications/
#author_profile: true
---

## Companny Projects
<hr style="border:2px solid gray">

### **XBRL Financial Tagging**
Built a **CNN** (Convolution Neural Network) model to map various sections (text paragraph) of directors report of a financial statements. Here, I automated the tagging of various paragraphs into suitable business compatible header by multi class classification using CNN. Also performed sentiment analysis to triage auditor's comments. 
I also contributed to build the framework for mapping system fields from client's financial data to tax filing template, using \textbf{Siamese-LSTM} architecture.

 ### **Sentence Similarity and Clustering**
 The task is to find similar ideas (given in text) provided by various participants in an innovation challenge. Here, I built similarity service after converting the ideas into **USE** (Universal Sentence Encoder) embedding. Also clustered the similar ideas using embedding matrix with the help of **Affinity Propagation**.
 
 ### **Idea to Reviewer Matching**
 The problem is to assign a set of reviewers for an idea submitted by a user in an innovation challenge. The reviewers have to be selected based on his/her competency/skills. So, given an idea the first task is to extract relevant skills (by which we can map reviewer) from a global set of skill-set to understand that idea. First, I used **RoBERTa** model to extract the embedding of submitted idea and mapped that to top-n skill set. Similarly **multilingual-USE** model is used to get another top-n skill set. For better precise result, intersection of those two skill-set is taken as final skills to understand that idea better. 
 
### **Online Learning**
 Built an online learning framework for sentence similarity task. Given a contribution, **USE** based similarity service displays a list of similar contributions in the production. Then the end user is asked to provide valuable feedback (star rating/up-down vote) regarding the quality of similarity result. The feedback from customer is automatically feed into the system and model is retrained to improve the performance. The whole pipeline is fully automatic and the latest fine tuned model is always deployed in the production without any manual intervention.
 
### **Recommendation System**
Built a deep learning based hybrid recommendation system which includes both content based and collaborative recommendation engine for algorithm recommendation.

### **Code Language Identification**
Built a framework to predict programming language of source code using a pre-trained deep learning model with accuracy more than 90%. 

## Masters Projects
<hr style="border:2px solid gray">

### **ADELE: Anomaly Detection from Event Log Empiricism**
Selecting diligent features from storage system log, we developed an effective machine learning model _ADELE_ for anomaly detection. The
model learns from system log history to assess the baseline of normal behavior and provide accurate and timely indications whenever something is amiss for the system. In this work we effectively highlights the challenges of the noisy log events and conducts an extensive log analysis to uncover the anomaly signatures. ADELE's capability to predict early paves way for online failure prediction for large scale systems.

### **GBTM: Graph Based Troubleshooting Method for Handling Customer Cases Using Storage System Log**
We developed an unsupervised troubleshooting methodology for identifying problematic modules responsible for system failure, thus reducing the complexity of the diagnostic process. First, we construct a sequence of time evolving dynamic graph from the collected system log of customer filed cases. We extract the initial set of problem creating modules identifying the anomalous substructure in the graph. In the second step, we extend this set of anomalous modules by detecting communities in the graph. Apart from providing troubleshooting modules, we rank the problem creating modules  which might be helpful for support engineers.

## Weekend Projects
<hr style="border:2px solid gray">
### :camera: :bookmark_tabs: Image Caption Generation

`Image Caption Generation` is a challenging task where a textual description is generated given a picture. It needs both methods from **Computer Vision** and **Natural Language Processing** to connect the image feature with words in the right order.


----

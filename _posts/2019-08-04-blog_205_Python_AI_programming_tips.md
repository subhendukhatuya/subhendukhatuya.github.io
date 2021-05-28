---
layout: post
title:  "Python Machine Learning Programming Tips, Best Practices"
date:   2019-08-04 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

# How to apply same data preprocessing steps to train and test data while working with scikit-learn?

The general idea is save the preprocessing steps in `.pkl` file using `joblib` and reuse them during prediction. This will ensure consistency. If you are using `scikit learn`, then there is an easy way to club preprocessing and modelling in same `object`.

Use `Pipeline`.

Example:

Say in your data you have both numerical and categorical columns. And you need to apply some processing on that and you also want to make sure to apply them during the prediction phase. Also both training and prediction phases are two different pipeline. In such situation you can apply something like this:

```py
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = data[feat_cols]
y = data["OUTCOME"]

numeric_features = feat_cols
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=42, stratify=y)

clf.fit(X_train, y_train)

clf.predict_proba(X_test)

model_file = "../model/model_randomforest.pkl"
joblib.dump(clf, model_file)

```

You can also add preprocessing steps for categorical column as well.

**Reference:**

- [Datascience stackexchange](https://datascience.stackexchange.com/questions/48026/how-to-correctly-apply-the-same-data-transformation-used-on-the-training-datas)



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>



----

# Machine Learning in Python with Scikit-Learn

Please follow [this](https://www.youtube.com/watch?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&time_continue=1565&v=irHhDMbw3xo&feature=emb_logo) quick refresher from youtube.

**Reference:**

- [Scikit-learn-tips](https://github.com/justmarkham/scikit-learn-tips)


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# How Python Manages Memory and Creating Arrays With `np.linspace()`

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How Python Manages Memory and Creating Arrays With np.linspace <a href="https://t.co/gRepVIQ35y">https://t.co/gRepVIQ35y</a></p>&mdash; PyCoder’s Weekly (@pycoders) <a href="https://twitter.com/pycoders/status/1343211790474358784?ref_src=twsrc%5Etfw">December 27, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 


----

# Refactoring Python Applications for Simplicity

- [link](https://realpython.com/python-refactoring/)


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Python CI/CD using Github action

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/V2TgkoExzvA" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/V2TgkoExzvA)_

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Pytoch gpu tuning

- [Object Detection from 9 FPS to 650 FPS in 6 Steps](https://paulbridger.com/posts/video-analytics-pipeline-tuning/)

<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Pro-tip for pytest users:

![image](https://pbs.twimg.com/media/Ej1PSMqWkAMK-Xa?format=jpg&name=small)


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


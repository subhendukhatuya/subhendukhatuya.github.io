---
layout: post
title:  "Build website using Jekyll and MathJax"
date:   2019-08-31 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

# Introduction

I write lot of content related to machine learning, deep learning. So obviously those contents have lots of math equation written in [latex](https://www.latex-project.org/). But all the popular blog hosting websties like [medium.com](https://medium.com/), [towardsdatascience.com](https://towardsdatascience.com/) etc don't support latex equation and I don't like them. Also I am a very lazy person. So my requirement is to write all the content, equation in [markdown](https://en.wikipedia.org/wiki/Markdown) file and push them to [Github](https://github.com/) and voila !! Content published in blog. Simple !!!  

But it's easier said than done. First of all Github doesn't render latex math equation by default in the markdown file. So check some package where you can build website using Markdown file. There are couple of such package but I prefer this particular one. Enter [Jekyll](https://jekyllrb.com/). And good part is Jekyll websites are easy to host in Github. 

So set up Jekyll, write content in markdown, push to Github and done. But only flaw is latex math equation rendering. So we need to configure the Jekyll with [MathJax](https://www.mathjax.org/). So below are the resources that will help you to set up a simple system like this. 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Create Sample Jekyll Blog and Host on Github

- [Youtube: Hosting Jekyll website on Github](https://www.youtube.com/watch?v=fqFjuX4VZmU)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Configuring Jekyll

Heavily borrowing from this [Blog](https://github.crookster.org/Adding-MathJAX-LaTeX-MathML-to-Jekyll/), it was a quick task to add. Add the following to Jekyll website repo:

```css
_includes/mathjax.html
assets/js/MathJaxLocal.js
_layouts/posts.html
```

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# mathjax.html

Copy the below snippet and put it in the above mentioned location.

```html
{% if page.mathjax %}
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script
  type="text/javascript"
  charset="utf-8"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
>
</script>
<script
  type="text/javascript"
  charset="utf-8"
  src="https://vincenttam.github.io/javascripts/MathJaxLocal.js"
>
</script>
{% endif %}
```

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# MathJaxLocal.js

Copy the below file and put it in the above mentioned repo.
- [MathJaxLocal.js](https://github.com/idcrook/idcrook.github.io/blob/master/assets/js/MathJaxLocal.js)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# posts.html

Put the below content in the above mentioned repo.

![image](/assets/images/post_html.png)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Final Repo Structure

```
├── 404.html
├── about.md
├── assets
│   └── js
│       └── MathJaxLocal.js
├── _config.yml
├── Gemfile
├── Gemfile.lock
├── _includes
│   └── mathjax.html
├── index.md
├── _layouts
│   └── post.html
├── _posts
│   ├── 2019-07-28-blog_000.md
│   ├── 2019-07-28-blog_001.md
├── pub.md
└── _site
```


<img src="https://images.squarespace-cdn.com/content/550c6978e4b0c1da40fd1208/1541615714652-2BCR3Y3BU5X4U7KYMT5R/that%27s+it+done+logo-01.png?format=1500w&content-type=image%2Fpng" alt="image" width="400"/>

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
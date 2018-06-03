---
layout: post
title: "Potatolemon - Planar Classification"
date: 2018-06-03 12:00:00 +0000
categories: open-source software
img: 20180603/title.png
---

This week, we continue with the cute but irrelevant title picture theme!

### Previously on Potatolemon...

1. [Logistic Activation Function](https://qichaozhao.github.io/potato-lemon-1/)
2. [Neuron Weights](https://qichaozhao.github.io/potato-lemon-2/)
3. [Layers and Networks](https://qichaozhao.github.io/potato-lemon-3/)
4. [Losses](https://qichaozhao.github.io/potato-lemon-4/)
5. [Backpropagation (Part 1)](https://qichaozhao.github.io/potato-lemon-5/)
6. [Backpropagation (Part 2)](https://qichaozhao.github.io/potato-lemon-6/)

---

# Planar Classification

Since we have ostensibly a working network, it's time to put it to the test with a further toy problem to learn!

So, in this post, we will look at a planar classification example which further tests our network in a binary classification setting.

The problem we will use is to try and classify gaussian quantiles apart from one another. When plotted on a 2d plane, here is what our classification problem looks like:

![Figure 1](/images/20180603/title.png)

Again, this is a non-linear problem (similar to the XOR problem we faced last time), and so requires at least one hidden layer to solve. In this case, we just use a network with one hidden layer (that's 5 Neurons wide), to see if we can learn to classify these points.

We first use our reference library (pyTorch) to train a network.

{: style="text-align:center"}
![Figure 2](/images/20180603/figure_1_pytorch_costs.png)

After training, we get the following results:

{: style="text-align:center"}
![Figure 3](/images/20180603/figure_2_pytorch_pred.png)

This looks pretty good - the boundary separates clearly the two classes, and the final accuracy for this model is 97%. We could probably improve this by increasing the size of our network if we wanted to.

Using potatolemon, we see the following training results:

{: style="text-align:center"}
![Figure 4](/images/20180603/figure_3_pl_costs.png)

This results in a very similar boundary.

{: style="text-align:center"}
![Figure 5](/images/20180603/figure_4_pl_pred.png)

The overall accuracy for our potatolemon network is 96%, so, fairly close performance wise (yet still different). Where do these differences come from? There are probably several sources of error, from the initialisation of the weights (which are drawn randomly from uniform distributions) to implementation details of the forward and backwards passes themselves (as even very minor numerical precision differences can produce large outcome differences over the course of many epochs). Also, let's not discount the fact that there still might be an elusive bug somewhere in potatolemon ;)

Aside from the result accuracy, the network took nearly twice as long as the pyTorch example to train. This is quite a large performance difference given the small size of the network where you would expect that the vectorisation and parallelism offered by pyTorch would not make such a huge difference.

These two minor points aside, the results of this test by and large are another validation of our network and the fact that it has been implemented correctly! Huzzah!

In the next post, we'll make a proper test of our network on the current hello world of neural network problems, the MNIST handwriting classification problem.

The code for the tests above can be found at: [https://github.com/qichaozhao/potatolemon/blob/master/examples/planar_classification.ipynb](https://github.com/qichaozhao/potatolemon/blob/master/examples/planar_classification.ipynb)

---

All the rest of the code can be found on the github project: [https://github.com/qichaozhao/potatolemon](https://github.com/qichaozhao/potatolemon)

Peace out for now!

-qz




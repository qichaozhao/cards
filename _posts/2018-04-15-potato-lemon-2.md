---
layout: post
title: "Potatolemon - Neuron Weights"
date: 2018-04-15 07:00:00 +0000
categories: open-source software
img: 20180415/title.png
---

### Previously on Potatolemon...

1. [Logistic Activation Function](https://qichaozhao.github.io/potato-lemon-1/)

---

# Neuron Weights 

Previously, we talked about how the Logistic function is a good basic way to classify a binary outcome (e.g. 0 or 1), and hence makes for a good basic activation function within our artificial model for a neural network.

We know conceptually speaking, that the other half of an artificial neuron has a bunch of weighted inputs that are summed and then fed into the activation function, so if we use a neuron that we model as follows:

![Figure 1](/images/20180415/figure_1_neuron.jpg)

Then, the equation for the output of the neuron will be as follows:

$$ y = f(w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n} + b) $$

In this equation we define the following quantities:

1. $$ y $$: the output of the neuron.
2. $$ f() $$: the activation function (the logistic function for now)
3. $$ w $$: a vector of weights applied to the input (shape of (x, 1))
4. $$ x $$: a vector of inputs (shape of (x, 1))
5. $$ b $$: a bias value that allows the activation function to be fitted better to the data.

Knowing that these are vectors, we can re-write using matrix multiplication as:

$$ y = f(w^{T}.x + b) $$

So, after not much math-ing, we have the entire equation that describes the neuron. Now we can go ahead and implement it.

# Implementation

We will implement the neuron as a class, so that when we build the network it will be composed of many Neuron objects. Each object can save its own weights, which will hopefully help us when it comes to implementing forward and backpropagation.

```python
"""
The Neuron class.

Holds its own weights and contains getters and setters for it.

Also contains the fire function, used to "activate" the neuron.
"""

import numpy as np

from .activations import sigmoid


class Neuron(object):


    def __init__(self, num_inputs, activation=sigmoid):

        self.activation = activation
        self.weights = np.random.rand(num_inputs, 1)
        self.bias = np.zeros(num_inputs)

    def forward(self, input):
        """
        In this function we implement the equation y = f(w^T . x + b)

        :param input: a column vector of shape (m, 1)
        :return: a column vector of shape (m, 1)
        """

        return self.activation(np.dot(self.weights.T, input))

    def get_weights(self):
        """
        Return the weights

        :return:
        """
        return self.weights

    def set_weights(self, weights):
        """
        Update the weights

        :return:
        """
        self.weights = weights
```

That's it for this post.

Next post, things start getting a bit more exciting as we will build some higher level abstractions (`Layers` and `Network`) that make it easier to construct a network out of individual neurons.

As with my previous post, all code can be found on the github project: [https://github.com/qichaozhao/potatolemon](https://github.com/qichaozhao/potatolemon)

Peace out for now!

-qz


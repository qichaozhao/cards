---
layout: post
title: "Potatolemon - Layers and Networks"
date: 2018-05-16 12:00:00 +0000
categories: open-source software
img: 20180515/title.jpg
---

### Previously on Potatolemon...

1. [Logistic Activation Function](https://qichaozhao.github.io/potato-lemon-1/)
2. [Neuron Weights](https://qichaozhao.github.io/potato-lemon-2/)
3. [Layers and Networks](https://qichaozhao.github.io/potato-lemon-3/)

---

# Losses

Previously on potatolemon, we finished building out the abstractions for the layer and the network that we'll rely on in the future to actually use our network with. This brought us to a point where we have a Multilayer Perceptron (or, a Neural Network that only goes in one direction).

Today, we'll talk about and implement the first part of training a Neural Network - the losses - because the network needs to know how much it is  wrong by if it is to begin learning. Losses can also be called costs or errors, and a loss function can also be called a cost function or error function.

To illustrate what we mean by a loss, let's think back to our logistic regression output from a few blog posts ago. Remember that for a single logistic function, we have a range of outputs between 0 and 1 (you can think of it as a range of probabilities that something might occur, but it's not quite that).

If we were to use this logistic function in a binary classification problem (i.e. perhaps trying to decide whether a picture is a cat or not), then we could use our training examples to compare our logistic function output against a ground truth that we already know. In this scenario, the difference between the two would be our loss.

Let's say for example that if a picture is of a cat, we will give it class label = 1, and if it is not, then we will give it a class label = 0. Then, we run this example through our logistic function. The loss function would then be able to be written as follows:

$$ loss = y_{true} - y_{pred} $$

- $$ y_{true} $$ is the label we have assigned (0 or 1)
- $$ y_{pred} $$ is the output of the forward pass (a probability between 0 and 1)

This equation defines the absolute loss (also known as the L1 loss), and is one of many different loss functions that we can apply to measure how well our prediction result matches up against the ground truth.

This L1 loss however is not the best loss metric to use for a classification problem - for that there is something called Cross Entropy. The equation given for the Average Cross Entropy loss is as below:

$$ loss = - \frac{1}{N} \sum_{n=0}^N \sum_{i=0}^I y_{ni} log (\hat{y_{ni}}) \qquad (1) $$

- $$ N $$ is the number of training examples.
- $$ i $$ is a class that is being predicted (of a total $$ I $$ classes)
- $$ y_{ni} $$ is the ground truth of class $$ i $$ for training example $$ n $$.
- $$ \hat{y_{ni}} $$ is the predicted value of class $$ i $$ for training example $$ n $$.

Although this seems more complicated than for our L1 loss, it is partly because we are firstly summing across all training examples and taking the average, and also because we have to deal with an arbitrary number of classes.

So, why is cross entropy a better loss metric to use L1 loss?

1. Cross Entropy loss penalises wrong classifications much more harshly than L1 loss.
2. Cross Entropy loss results increases the effectiveness of training a neural network due to how the equation differentiates during backpropagation - it reduces the vanishing gradient problem compared to L1 loss.

For proving point 1, we can take a look at the figure below, which plots 
 the Cross Entropy loss versus L1 loss for varying predictions vs a ground truth of 1.
 
 ![Figure 1](/images/20180515/figure_1_loss_comparison.png)

You can see immediately that if our ground truth is `1`, we get an exponentially increasing penalty the further away we are if we use log loss (up to an undefined value if we actually predict `0`), versus just a linearly increasing penalty (up to a maximum of `1`).

For proving point 2 - we will tackle this in a future blog post when we implement the backpropagation algorithm.

So, knowing that we will now choose the Average Cross Entropy, we actually need to end up implementing two variants. One for binary classification and one for multi-class classification.

For a multi class classification, i.e. `I > 2`, then we end up having one output node per class, which means we can use the Average Cross Entropy loss equation given above.

For a binary classification though, we actually end up just using one output node to represent the two possible outcomes. If the output node is zero then that gives us the first class, and if the output node is one, then that gives us the second class. This results in a slightly different Average Cross Entropy equation that we can derive from the above.

First, we set $$ I = 2 $$ and expand the summation term over all $$ i $$.

$$ loss = - \frac{1}{N} \sum_{n=0}^N y_{n1} log (\hat{y_{n1}}) + y_{2n} log (\hat{y_{2n}}) $$

Next, we recognise that since we are using one output node to represent two states, the state $$ y_{2n} $$ can be re-written in terms of $$ y_{1n} $$. This is because if one state is 1, then the other must be 0.

$$ y_{n2} = 1 - y_{n1} $$

Now we can substitute this directly into the equation:

$$ loss = - \frac{1}{N} \sum_{n=0}^N y_{n1} log (\hat{y_{n1}}) + (1 - y_{n1}) log (1 - \hat{y_{n1}}) \qquad (2) $$

And so we get our Average Binary Cross Entropy equation.

Now, all we are left to do is the relatively simple matter of translating our equations `(1)` and `(2)` into code.

```python
"""
This file contains all of our cost functions.
"""

import numpy as np


def binary_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the cross entropy loss for a binary classification output (e.g. one output node).

    :param pred: A vector value that is the output from the network of shape (training_examples, 1)
    :param actual: A vector value that is the ground truth of shape (training_examples, 1)
    :return: A vector value representing the binary_crossentropy loss of shape (training_examples, 1)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    m = pred.shape[0]
    J = -1 / m * np.sum(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))

    return J


def categorical_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the average entropy loss for a multi class classification output (e.g. multiple output node).

    :param pred: A vector value that is the output from the network of shape (training_examples, num_classes)
    :param actual: A vector value that is the ground truth of shape (training_examples, num_classes)
    :return: A vector value representing the binary_crossentropy loss of shape (training_examples, 1)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    print(pred)

    m, i = pred.shape

    # We loop through each class and calculate our final cost
    # This could probably be done in a quicker fashion via a matrix multiplication but for simplicity's sake we use a loop
    J = 0
    for cls in range(i):

        pred_cls = pred[:, cls]
        actual_cls = actual[:, cls]

        J = J + (- 1 / m * np.sum(actual_cls * np.log(pred_cls)))

    return J
```

If you read through the code above, you'll notice I add in an extra few lines for slightly perturbing prediction values away from absolute 1s and 0s (although these are very unlikely anyway). This must be done to avoid `inf` values being returned by the `np.log` function in case our predictions do end up being 1 or 0.

Okay, so that's all for this post! Slightly more mathsy than we've had previously, but I hope it all made sense.

Next post we tackle probably the most interesting part of building a neural network - the backpropagation and weight update. This is what allows the networks to train!

---

All code can be found on the github project: [https://github.com/qichaozhao/potatolemon](https://github.com/qichaozhao/potatolemon)

Peace out for now!

-qz


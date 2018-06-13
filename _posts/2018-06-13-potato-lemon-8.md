---
layout: post
title: "Potatolemon - Multiclass Classification"
date: 2018-06-13 12:00:00 +0000
categories: open-source software
img: 20180613/title.jpg
---

### Previously on Potatolemon...

1. [Logistic Activation Function](https://qichaozhao.github.io/potato-lemon-1/)
2. [Neuron Weights](https://qichaozhao.github.io/potato-lemon-2/)
3. [Layers and Networks](https://qichaozhao.github.io/potato-lemon-3/)
4. [Losses](https://qichaozhao.github.io/potato-lemon-4/)
5. [Backpropagation (Part 1)](https://qichaozhao.github.io/potato-lemon-5/)
6. [Backpropagation (Part 2)](https://qichaozhao.github.io/potato-lemon-6/)
7. [Planar Classification](https://qichaozhao.github.io/potato-lemon-7/)

---

Previously on potatolemon, we continued testing our neural network by running it on a binary classification problem, and discovered it performed more or less similarly to a reference implementation in pyTorch!

# Multiclass Classification

Now, we perform the final testing on our library to see if it can also learn a multiclass classification problem!

In this post, we look at the classic Iris dataset that can be loaded directly from the `sklearn` library. This is a multiclass classification problem - the aim being to learn to classify 3 different iris flower species based on 4 different features.

We first use our reference library (pyTorch) to train a network.

{: style="text-align:center"}
![Figure 2](/images/20180613/figure_1_pytorch_costs.png)

After training, we get the following results:

{: style="text-align:center"}
![Figure 3](/images/20180613/figure_2_pytorch_cm.png)

This looks more or less okay - the confusion matrix tells us that the network can separate the 3 species fairly well. If we look at a classification report, we get the following:

```python
             precision    recall  f1-score   support

          0       1.00      1.00      1.00         4
          1       0.64      1.00      0.78         9
          2       1.00      0.29      0.44         7

avg / total       0.84      0.75      0.71        20
```

An overall F1-score of 0.71 is pretty average, and it is significantly worse than an out of the box random forest classifier (which scores 0.84).

However, the point here is not to completely smash the prediction problem (which we could do if we used a bigger/deeper network), but to see how our potatolemon library stacks up.

So, using potatolemon, we see the following training results:

{: style="text-align:center"}
![Figure 4](/images/20180613/figure_3_pl_costs.png)

With a confusion matrix as:

{: style="text-align:center"}
![Figure 5](/images/20180613/figure_4_pl_cm.png)

With a classification report of:

```python
             precision    recall  f1-score   support

          0       1.00      1.00      1.00         4
          1       0.73      0.89      0.80         9
          2       0.80      0.57      0.67         7

avg / total       0.81      0.80      0.79        20
```

These results whilst not great from an ultimate prediction accuracy point of view, are fantastic for the library implementation, as they show that our potatolemon network behaves in roughly the same way as a pyTorch network implemented as a reference - this verifies that we have implemented everything succesfully! Hooray!

It is worth noting that for both networks they were very sensitive to the initialisation conditions - on certain random seeds the networks would get stuck in local optima and be unable to learn the dataset fully, whereas on others the results would be very very good (F1-scores of 0.95 or so), but for fairly small networks on a small problem set this is an expected issue.

Also, even on this small scale (3 layers, 10 neurons wide), the pyTorch library starts to show its advantages performance wise. Training took ~7 seconds, compared to ~26 seconds using potatolemon. But again, that is to be expected.

The important thing is that the networks appear to learn at similar rates, and achieve similar results!

The code for the tests above can be found at: [https://github.com/qichaozhao/potatolemon/blob/master/examples/multiclass_iris.ipynb](https://github.com/qichaozhao/potatolemon/blob/master/examples/multiclass_iris.ipynb)

---

Okay. Phew.

It has been a fun ride the last few months building this library, and it is immensely pleasing to arrive at a nominally working product!

I do intend to keep expanding this library in the future - there are still many areas of neural networks I would like to tackle on an implementation level - CNNs, RNNs, LSTMs and GANs, and these will come over time.

For now though, I will take a break from the potatolemon project to pursue some other side projects which I will write about in the next post!

Till then then, my friends.

-qz




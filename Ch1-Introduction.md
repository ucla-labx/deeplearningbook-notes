# Chapter 1: Introduction

True challenge of AI is to perform easy tasks that are hard to formally describe. 

Artificial intelligence to solve intuitive problems is best approached via ML.

Abstract and formal tasks are hard for humans but easy for computers (chess).
Choice of representation has a huge effect on ML problems.

Representation learning learns the representation of the problem by itself - autoencoders, MLP

MLP - hidden layers and output layer.
    + learns intermediate representations of the data that are progressively more abstract.

Can measure depth in one of two ways - as a function of the computational graph (depending on stuff) or as a function of the number of abstractions (# of hidden layers).

Machine learning is the only viable approach to building AI systems according to the authors of this books.

Great power and flexivility in deep learning.

Historical Deep learning.
    + Deep learning has improved with more training data, computational power.
    + ADALINE (1950s) used stochastic gradient descent - batches.  - linear models.
    + **linear models** struggle with XOR function.
    + neuroscience is a source of inspiration.
    + ReLU (rectified linear unit) is used to capture nonlinearity.

## Connectionism
**connectionism** arose from trying to describe the mind in terms of symbolic reasoning.

This introduced the idea of a **distributed representation** that each input to a suystem should be represented by many features and each feature should be involved in the representation of many possible inputs. i.e. learn r,g,b independently of car,truck,bike.

Another accomplishment of this period was back-propogation. 

1990s - modelling sequences via LSTMs,

Kernel machines and graphical models took over after this. 

2006 - Geoffery Hinton deep belief network with more efficient training and larger datasets with greedy layerwise training.

Dataset sizes and Model sizes have increased greatly (Doubled every 2.4 years)

ImageNet 2012 - 26.1% to 15.3% to 3.6%

LSTM are handling sequences and 
neural Turing Machines are going crazy

**reinforcement learning** is being used to solve more and more tasks.

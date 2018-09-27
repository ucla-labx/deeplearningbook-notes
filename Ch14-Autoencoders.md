# Chapter 14: Autoencoders

An autoencoder consists of two parts.
- An encoder $$f(x)$$ that maps some input representation $$x$$ to a hidden, latent representiation $$h$$, 
- A decoder $$g(h)$$ that reconstructs the hidden layer $$h$$ to get back the input $$x$$

We usually add some constraints to the hidden layer - for example, by restricting the dimension of the hidden layer. An **undercomplete autoencoder** is one where $$dim(h) < dim(x)$$. 

By adding in some sort of regularization/prior, we encourage the autoencoder to learn a distributed representation of $$x$$ that has some interesting properties.

We train by minimizing the $$L^2$$ loss between the input and output. It's important to note for an autoencoder to be efffective, we need some sort of regulazation/constraint - otherwise it's trivially simple for the autoencoder to learn the identity function. 

The autoencoder is forced to consider these two terms - minimizing the regularazation cost as well as the reconstruction cost. In doing so, it learns a hidden representation of the data that has interesting properties.

### Sparse Autoencoder

$$L(x, g(f(x))) + \Omega(h)$$ 

As always, we can train this model using MLE by maxamizing $$\log p(h,x)$$, which corresponds to maxamizing $$\log p(h) + \log p(x \vert h)$$

We can think of these two terms as the regularization and reconstruction cost respectively. 

For example, for sparse autoencoders, $$p(h)$$ is the Laplacian distribution $$\frac{\lambda}{2} e^{- \lambda \vert h_i \vert}$$. This distribution corresponds to the absolute value sparsity penalty (L1 regularization).

For a $$L^2$$ regularization penalty, $$p(h)$$ is a Gaussian distribution.

### Denoising Autoencoder

Here we want to minimize the loss function $$L(x, g(f(\tilde{x})))$$, where $$\tilde{x}$$ is $$x$$ with some corruption or noise. 

This corresponds to minimizing $$E_{x~ \hat{p}_data} E_{\tilde{x}~ L( \tilde{x} \vert x)} \log p_decoder (x \vert h)$$  where $$h = f(\tilde{x})$$.

Alterative we can use score matching as an alterative to maximizing the log likelihood - which learns the gradient field.

Denoising autoencoders (DAE) learn the vector field of $$g(f(x)))$$ which we can use to estimate the score of the data distribution.

With a regularization loss $$\Omega (h) = \lambda \Vert\frac{\partial f(x)}{\partial x}\Vert^2_F$$, a denoising autoencoder learns to not respond to small changes to it's input - it esssentially learns how to map the corrupted example back onto the manifold - this is what gives us our vector field.

We could use this to train models to colorize pictures, deblur images, etc.

### Learning Manifolds

We aim to learn the structure of the manifold through the distribution representation learned by the autoencoder. 

The manifold has tangent planes (similar to tangent lines). These tangent lines describe how to move along the manifold. 

1. We need to learn a representation $$h$$ such that we can reconstruct $$x$$
2. We need to satisfy constraint/regularization penalty.

As such, we learn variations along the manifold - we need to know this because we must remap onto the manifold in the case of a denoising autoencoder.

Most early research was based on nn approaches - we try to solve a linear system to get "pancakes", tying these pancakes togethere to form a global system. 

However, if manifolds vary a lot, then we need a lot of "pancakes" to capture this variation - so instead we look towards deep learning to solve these problems.

### Contractive Autoencoders

Contractive autoencoders trained with sigmoidal units createa  sort of binary code in their hiddne layer. 

The cost function approaches f(x) as a linear operator. In this instance the term contractive comes from the fact that $$norm(J) \leq 1 \forall x s.t. \Vert{x}\Vert=1$$. We are essentially shrinking, or contracting the unit sphere. 

Usually, only a small number of hidden units have large derivatives - these corresponds to movement along the manifold, as the capture most of the variance of the data. 

One thing we can do to avoid learning trivial transformations is to tie the network weights together.

### Predictive Sparse Decomposition

This is a hybrid of sparse and parametric autoencoders, where we try to 
$$ min \Vert x-g(h)\Vert^2 + \lambda \vert h \vert_1 + \gamma \Vert h-f(x)\Vert^2$$



We can use autoencoders to perform a variety of tasks - such as dimensionality reduction. 

If we learn a binary coding, we can use this coding to determine similarity, to do a form of **semantic hashing**, where we can use the Levinghstam distance of two codes to determine their similarity.



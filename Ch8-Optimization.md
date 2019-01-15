# Ch8: Optimization for Deep Learning

#### Introduction
- Overall goal: find parameters $\theta$ of a neural network that optimize a defined cost function $J(\theta)$
  - $J$ is generally some quantification of our "performance" over teh training data as well as additional regularization terms
- In actuality, we'd like to minimize $$J(\theta) = \mathbb{E}_{(x,y) ~ p_{data}} L(f(x;\theta), y)$$ 
  - This is the loss across the entire data generating distrbution, and is intractable, since we don't have access to the data generating distribution
  - This quantity is also known as the "risk"

#### Empirical Risk Minimization

- We approximate the above quantity by minimizing across the training data: $$\frac{1}{m}\sum_{i=1}^{M} L(f(x;\theta), y)$$ 
- Often, we can't even use empirical risk minimization directly, due to our loss function $L$ being infeasible to optimize - such as the 0/1 loss function.
- To work around this, a **surrogate loss function** is often used, such as the negative log-likelihood, as an approximation for the 0/1 loss.
- We can also compute the 0/1 loss across validation data during training, and when it stops decreasing it may be an indication for early stopping.

#### Batch/minibatch learning

- If we formulate our loss and gradient as the expectation across the # of training examples, this will be expensive, due to large training set sizes
- Using batches of smaller size is faster, and provides many advantages:
  - Accuracy of the gradient is not too much worse- the standard error of the mean is $\frac{\sigma}{\sqrt(n)}$  , meaning that there is less than a linear improvement in the gradient accuracy with more samples. 
  - Minibatch learning accounts for redundancy and introduces more stochasticity into the  model, has a regularizing effect
  - If examples are not repeated, then it can be shown that minibatch SGD provides an unbiased estimate of the gradient of the *generalization error* - see pg. 278 of the DLB for a derivation of this.
- If you have smaller batches, you may want to use a smaller learning rate, since the gradient has more variability, and you are "less confident" about taking a larger step in that direction.

#### Issue w/Optimization: Ill-Conditioning

- Said to happen when the Hessain $H$ is large compared to the gradient norm, and even a very small step in the direction of the gradient would actually increase the cost function rather than decreasing it as desired
- How to find ill conditioning: the Taylor series expansion. Recall that we can have a 2nd order approximation of $f$ at $x_0$: $f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2$. If we let $\textbf{g}$  denote the first derivative and $\textbf{H}$ denote the 2nd derivative, then we have $f(x)\approx f(x_0) + (x-x_0)\textbf{g} + \frac{1}{2}(x-x_0)^T\textbf{H}(x-x_0)$. where $x$ and $x_0$ are now vector-valued.

#### Issue w/Optimization:  Saddle Points/Local Minima

- Since neural networks lead to the optimization of highly nonconvex loss functions, optimization algorithms could potentially converge to a local minima with a significantly higher cost than the global minima
- Can test if  local minima is a problem in model training by plotting the norm of the gradient —> it was shown that local minima is usually not the problem, since the norm of the gradient is still quite large when convergence occurs, whereas if it was local minima then the gradient would be about 0. 
- **Saddle points** are points with $0$ gradient, but these points are neither a local min or local max
- We can use a generalization of the 2nd derivative test through examining the eigenvalues of the Hessian
- When $\nabla_xf(x) = 0$ and the Hessian is positive definite (all eigenvalues > 0), then we're at a local min, since the 2nd directional derivative is everywhere positive, and the grad is 0
- Similarly, when the Hessian is negative definite (all eigenvalues < 0) then we're at a local max
- If it's neither, then we're at a saddle point
- Notion of **convexity** - central to traditional machine learning loss functions - the Hessian needs to be everywhere positive semidefinite, meaning that we should be able to show for a Hessian $H$ and any real vector $z$, $z^T H z > 0$. 
- Saddle points are generally problematic for 2nd order optimization algorithms - such as Newton's method, which is only designed to jump to a point with 0 gradient, so it may jump to a saddle point
  - Modifications such as "Saddle-free Newton's method" exists to accomodate for this

#### Issue w/Optimization: Cliffs/Exploding Gradients

- The loss function surface may have large step-function like increases, leading to large gradients. But since the gradient is generally used for the direction of steepest descent and not necessarily the amount to step, we can accomodate for large gradients by imposing a gradient clipping rule. This preserves direction but decreases magnitude.

#### Issue w/Optimization: Long-term dependencies

- The vanishing/exploding activations/gradient problem:
  - Consider a deep feedforward network that just successively multiplies its input by a matrix $W$ $t$ times. If $W$ has an eigendecomposition $W = V\mathbb{diag}(\lambda)V^{-1}$ then $W^t = V\mathbb{diag}(\lambda)^tV^{-1}$, so unless the eigenvalues are all around an absolute value of $1$, we will have the vanishing/exploding activation problem, and the gradient propagating backwards scales similarly. 

#### Issue w/Optimization: Poor Correspondence Between Local & Global Structure

- Being initialized on "the wrong side of the mountain":

  ![](images/mountain.png)

- Many loss functions don't have an absolute difference - consider the softmax loss function that measures the cross entropy between the labels' distribution and your predictions' distribution - it can never truly reach 0, but it can get arbitrarily close as the classifier gets more and more confident about its predictions.

- Overall, its important to try to choose a good initialization scheme, and possibly try several different random initializations, train the model, and pick the one that did the best

  ​




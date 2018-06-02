# Ch7: Regularization for Deep Learning

#### Introduction
- Crux: How to make machine learning algorithms perform well on not just data its been trained on, but also inputs that it has never seen before
- Regularization is defined as anything that's done to decrease an algorithm's test error at the expense of its training error. 


Most deep learning regularization strategies try to trade increased bias for reduced variance (Remember that MSE can be decomposed into a bias term and a variance term)

#### Strategies for Regularization
- Put constraints on the model: adding restrictions on parameter values (such as a max norm)
- Add extra terms to the objective function, which can be thought of as a soft constraint on the model parameters
- A model that has overfit is said to have learned the data generating process but also many other generating processes, ie a model that has low variance but high bias. With regularization, we aim to take this model and regularize it to become a model that matches the data generating process.


#### Paramter Norm Penalites

We can try to limit the capacity of models by adding a penalty term $\Omega(\theta)$ to the cost function.
$$ \tilde{J}(\theta, X, y) = J(\theta, X, y) + \alpha \Omega(\theta)$$

The $\alpha$ term corresponds to the relative contribution of the regularization penalty to the overall cost/loss.

One thing to note is that we usually use $w$ to denote all the weights that should be affected by a norm penalty, while $\theta$ denotes all parameters. We usually do not regularize the bias, since it generally has a lower variance than the weights, due to the bias not interacting with both the inputs and the outputs, as the weights do.

For example for a linear model $f(x) =mx+b$ the bias term allows us to fit lines that do not pass through the origin - however if we do weight deacy on the bias term, we are encouraging the bias to stay close to 0, which defeats the purpose of the bias term in the first place.

- We can also consider using a different parameter norm penalty $\alpha$ for each layer in the neural network, but due to the additional complexity this introduces in searching for the optimal hyperparameters, the norm penalty is generally the same for each layer.


#### L2 Regularization

One of the simplest regularization techniques is **weight decay**, where $\Omega(\theta) = \frac{1}{2} \Vert w \Vert_2^2$. This is also known as **ridge regression** or **Tikhonov regularization**.

$$ \tilde{J}(\theta, X, y) = J(\theta, X, y) + \frac{\alpha}{2} w^Tw $$

Taking the gradient of this cost function yields

$$\nabla_w \tilde{J}(\theta, X, y) = \alpha w + \nabla_w J(\theta, X, y) $$

So for a single gradient step, $$ w \leftarrow (1- \epsilon \alpha)w - \epsilon \nabla_w J(w; X,y) $$.

#### Effect of L2-Regularization on parameters learned

- We can use a quadratic approximation of our cost function $J(w)$. This is the second-order Taylor series expansion. In a single dimension, this is something like $$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$$.

- To understandthe effect of l2 regularizationriz on the parameters learned, we can approximate $J$ around the optimal weights $w*$: $$J(\theta) = J(w^*) + \nabla J(w^*)(w-w^*) + \frac{1}{2}(w-w^*)^T\textbf{H}(w-w^*)$$
- Since w* are our optimal weights that minimize the cost function, the second term is eliminated, as $\nabla J(w^*) = 0$. 

- Thereofre, the minimum of our approximation $\hat{J}$ occurs when $\nabla_w \hat{J}(w) = H(w-w*) = 0$. If we now consider a regularized version of the approximation, we have to add the gradient of the regularization penalty $\frac{\alpha}{2}w^Tw$ to the minimization objective: $$\alpha w^* + H(w-w^*) = 0$$ Giving us $$\tilde{w} = (H + \alpha I)^{-1}Hw^*$$.
- If we do an eigendecomposition on H, letting $H = Q\Lambda Q^T$ where $\Lambda$ is a diagonal matrix who's diagonal entries are eigenvalues of $H$ and $Q$ is a matrix who's columns are eigenvectors of $H$ that form an orthonormal basis. We obtain $\tilde{w} = Q(\Lambda + \alpha I)^{-1}\Lambda Q^T w*$.
- This means that the effect of the $L^2$ weight decay is to rescale $w$ along the axes defined by the Hessian of the cost function $H$. 
- Specifically, when $\lambda_i >> \alpha$, the regularization effect is small: this means that in directions where the second order derivative of the cost function $J$ is large, meaning that the cost function has high curvature in that area, the regularization effect will be small, if any. On the other hand, in directions where the eigenvalues of $H$ are small, the regularization effect will be large.
- This intuitively makes sense because we want to penalize the weights in directions that do not have a high curvature (and thus do not contribute significantly to reducing the objective), so we have a high reguarlization penalty. On the other hand, for weights in the directions of high curvature, we do not regularize those weights as much since they contribute significantly to reducing the overall cost function.


- For linear regression, adding in L2 regularization alters the normal equation solutions for $w$ from 

$$w = (X^TX)^{-1} XTy$$ 

to $$w = (X^TX + \alpha I)^{-1} XTy$$. This makes linear regression shrink weights on features whose covariance is low compared to the added variance $\alpha I$

#### L1 Regularization
- L1 regularization places a penalty on the absolute values of the weights rather than their squared norm as L2-regularization does. It is defined as $$\Omega(w) = \sum_i (\vert w_i \vert)$$.
- The regularized objective is $\tilde{J}(w) = \alpha \Vert w \Vert_1 + \nabla_w J(X, y;w)$

This leads to a gradient $$\nabla_w \tilde{J}(w) = \alpha \text{sign}(w) + \nabla{w}J(X,y ;w)$$

We can see that the regularization contribution does not scale linearly with $w_i$ but its effect is dependent on the sign of the weight. 

$$\nabla_{w_i}J = H_{i,i}(w_i-w_i^*) + \alpha \text{sign}(w_i) $$

To solve for $w_i$, we set the gradient to be 0
$$0 =  H_{i,i}(w_i-w_i^*) + \alpha \text{sign}(w_i) $$

Solving for $w_i*$, we get
$$w_i^* = w_i + \frac{\alpha}{H_{ii}}\text{sign} (w_i)$$

Since $\alpha$ and $H_{ii}$ are both strictly >0
In the case when $w_i$ is positive $w_i$ + $\frac{\alpha}{H_{ii}}$ is positive. 
In the case when $w_i$ is negative $w_i$ - $\frac{\alpha}{H_{ii}}$ is negative. 
When $w_i$ is 0, we get $w_I^*$ is 0 as well.
Thus, $$\text{sign} (w_i^*) = \text{sign}(w_i)$$




This gives us the following expression for $w_i$: $$w_i = \frac{-\alpha}{H_{ii}}\text{sign}(w_i*) + w_i*$$

Using the property $\frac{w_i^*}{\text{sign}(w_i^*)} = \vert w_i^* \vert$

$$w_i = \text{sign}(w_i^*) (\vert w_i^*\vert - \frac{\alpha}{H_{ii}})$$

However, this is not yet complete: if $(\vert w_i^*\vert - \frac{\alpha}{H_{ii}}) < 0$, then the sign of $w_i$ will get flipped: it will no longer take the sign of $w_i$. Since we've shown above that $sign(w_i*) = sign(w_i)$, we enforce that the quantity cannot be less than $0$:

$$w_i = \text{sign}(w_i^*) \max(0,(\vert w_i^*\vert - \frac{\alpha}{H_{ii}}))$$

#### Intepretation of L1 Regularization

- This means that in the case where $w_i* \leq \frac{\alpha}{H_{ii}}$, the regularized learner sets $w_i = 0$, otherwise $w_i$ is shifted by $\frac{\alpha}{H_{ii}}$. 
- L1 regularization results in sparse weights being learned, meaning that some of teh parameters hve their optimal value set to be 0. 
- Therefore, L1 regularization can be considered as doing some sort of feature selection: the nonzero parameters indicate what features should be used. 
- L1 regularization is equivalent to doing MAP estimation (basically MLE estimation with a prior on your weights) using a Laplacian prior, while L2 regularization is equivalent to imposing a Guassian prior on your weights.

...To be continued.


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

So for a single gradient step,

$$ w \leftarrow (1- \epsilon \alpha)w - \epsilon \nabla_w J(w; X,y) $$

#### Effect of L2-Regularization on parameters learned

- We can use a quadratic approximation of our cost function $J(w)$. This is the second-order Taylor series expansion. In a single dimension, this is something like $$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$$.

- To understandthe effect of l2 regularizationriz on the parameters learned, we can approximate $J$ around the optimal weights $w*$: $$J(\theta) = J(w^*) + \nabla J(w^*)(w-w^*) + \frac{1}{2}(w-w^*)^T\textbf{H}(w-w^*)$$
- Since w* are our optimal weights that minimize the cost function, the second term is eliminated, as $\nabla J(w^*) = 0$. 

- Thereofre, the minimum of our approximation $\hat{J}$ occurs when $\nabla_w \hat{J}(w) = H(w-w*) = 0$. If we now consider a regularized version of the approximation, we have to add the gradient of the regularization penalty $\frac{\alpha}{2}w^Tw$ to the minimization objective: $$\alpha w^* + H(w-w^*) = 0$$ Giving us $$\tilde{w} = (H + \alpha I)^{-1}Hw^*$$.
- If we do an eigendecomposition on H, letting $H = Q\Lambda Q^T$ where $\Lambda$ is a diagonal matrix who's diagonal entries are eigenvalues of $H$ and $Q$ is a matrix who's columns are eigenvectors of $H$ that form an orthonormal basis. We obtain $\tilde{w} = Q(\Lambda + \alpha I)^{-1}\Lambda Q^T w*$.
- This means that the effect of the $L^2$ weight decay is to rescale $w$ along the axes defined by the Hessian of the cost function $H$. 
- Specifically, when $\lambda_i >> \alpha$, the regularization effect is small: this means that in directions where the second order derivative of the cost function $J$ is large, meaning that the cost function has high curvature in that area, the regularization effect will be small, if any. On the other hand, in directions where the eigenvalues of $H$ are small, the regularization effect will be large.
- This intuitively makes sense because we want to penalize the weights in directions that do not have a high curvature (and thus do not contribute significantly to reducing the objective), so we have a high reguarlization penalty. On the other hand, for weights in the directions of high curvature, we do not regularize those weights as much since they contribute significantly to reducing the overall cost function.


- For linear regression, adding in L2 regularization alters the normal equation solutions for $w$ from $$w = (X^TX)^{-1} XTy$$ to $$w = (X^TX + \alpha I)^{-1} XTy$$. This makes linear regression shrink weights on features whose covariance is low compared to the added variance $\alpha I$

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

Using the property $\frac{w_i^* }{ \text{sign} (w_i^* )} = \vert w_i^* \vert$

$$w_i = \text{sign}(w_i^*) (\vert w_i^*\vert - \frac{\alpha}{H_{ii}})$$

However, this is not yet complete: if $(\vert w_i^* \vert - \frac{\alpha}{H_{ii}}) < 0$, then the sign of $w_i$ will get flipped: it will no longer take the sign of $w_i$. Since we've shown above that $sign(w_i*) = sign(w_i)$, we enforce that the quantity cannot be less than $0$:

$$w_i = \text{sign}(w_i^*) \max(0,(\vert w_i^*\vert - \frac{\alpha}{H_{ii}}))$$

#### Intepretation of L1 Regularization

- This means that in the case where $w_i* \leq \frac{\alpha}{H_{ii}}$, the regularized learner sets $w_i = 0$, otherwise $w_i$ is shifted by $\frac{\alpha}{H_{ii}}$. 
- L1 regularization results in sparse weights being learned, meaning that some of teh parameters hve their optimal value set to be 0. 
- Therefore, L1 regularization can be considered as doing some sort of feature selection: the nonzero parameters indicate what features should be used. 
- L1 regularization is equivalent to doing MAP estimation (basically MLE estimation with a prior on your weights) using a Laplacian prior, while L2 regularization is equivalent to imposing a Gaussian prior on your weights.


## Norm Penalties as Constrained Optimization

The generic parameter norm regularized cost function is of form 

$$ \tilde{J} (\Theta; X, y) = J(\Theta; X, y) + \alpha \Omega(\Theta)$$

We can minimize a function subject to constraints by constructing a generalized Lagrange function, which consists of the original objective function plus a set of penalties. 

Each penalty is a product between a KKT multiplier and a function representing whether the constraint is satisfied.

$$ \mathcal{L} (\Theta, \alpha; X, y) = J(\Theta; X, y) + \alpha(\Omega(\Theta) - k)$$

Here $\alpha$ is the KKT multiplier and $k$ is the max value of the norm of $\Theta$. 

The solution to this constriaed optimization problem is given by

$$\Theta^* = \underset{\Theta}{\operatorname{argmin\,}} \underset{\alpha, \alpha \geq 0}{\operatorname{argmax\,}} \mathcal{L} (\Theta, \alpha)$$ 

Whenever $\Omega(\Theta) > k$ $\alpha$ will increase, and whenever $\Omega(\Theta) < k$ $\alpha$ will decrease. The optimal value $\alpha^*$ will encourage the norm to shrink, but not so strongly that it becomes less than $k$.

If we fix $\alpha^*$, we can see that

$$\Theta^* = \underset{\Theta}{\operatorname{argmin\,}} \mathcal{L} (\Theta, \alpha^*) = \underset{\Theta}{\operatorname{argmin\,}} J(\Theta; X, y) + \alpha^*\Omega(\Theta) $$

This is the exact same cost function described earlier. Thus we can view norm penalties as constrained optimization. $L^2$ regularization constrains the weights to lie on a $L^2$ ball. 

The exact size of the constrained region is dependent on $J$, so we don't know this value exactly - however, we can control it rougly by altering $\alpha$. A larger $\alpha$ means that the regularization loss contributes more to the cost function, which corresponds to a smaller constrained region. 

We may want to use explicit constraints instead of penalties, as we can project $\Theta$ back onto the nearest point in the constrained region, which is useful if we have a general idea of $k$ already. 

In addition, we may get stuck in local minimia while training with norm penalties, which usually correspond to small $\Theta$. Explicit constraints implemented by reprojection will only affect the weights when they become large and attempt to leave the constrained region, allowing gradient descent to find a better local minimia inside the region.

Furthermore we can avoid large oscillations associated with a high learning rate. It is recommended to constrain the norm of each column of the weight matrix, which ensures that no singular hidden unit from having large weights.

### Regularization and Under-Constrained Problems

Many machine learning methods require inverting the matrix $X^TX$, which is not possible when $X^TX$ is singular - for example when there is no variance observed in some direction. 

In this case we can add regularization , which corresponds to inverting the matrix $X^TX + \alpha$ instead. 

This concept of using regularization to solve underedetermined linear equations extends beyond machine learning. For example, the **Moore-Penrose** pseudoinverse described in Chapter 2 is:

$$ X^+ = \lim_{a \to 0} (X^TX + \alpha I)^{-1} X^T $$

This is just linear regression with weight decay. 

One particular instance of an under-constrained problem is when we conduct linear regression on a one-hot encoded categorical feature vector. In this case, without regularization, we must normalize the data by fixing one of the classes to be all 0.

### Dataset Augmentation

The best way to improve model performance is with more data. Sometimes, we can create fake data and add it to the dataset to make our model better. This has been particularily effectve for object recognition. 

We can modify the base images with some small translation, or rotation. This approach makes the network learn weights that are robust to these transformations, On occasion, we can inject a bit of noise to the inputs, or even the hidden layers. When comparing two different algortims, we must compare there performance on similar datasets, which can be a subjectvie matter.

### Noise Robustness

For some models, adding some noise to the input is the same thing as a norm penalty. Injecting noise can be especially effective when it's applied to the hidden units.

This is a central concept of the denoising autoencoder. Occasionally, we can also add some small random noise to the weights as well, which serves as a stochastic implementation of Bayesian inference. This also encourages th e model to converge into regions that are not only minimia but minimia that are surrounded by flat regions. 

For small $\nu$ minimizing a cost function $J$ with added weight noise $\nu \textbf{I}$ is equivalent to minimizing the same cost function with an added regularization term $\nu E[\Vert \nabla_W \hat{y}(x) \Vert^2]$.

In most datasets there is some set of mislabeled data which may make MLE training harmful. We can explicitly model this by using noisy labels. Furthermore we can use **label smooothing**, which replaces the hard 0 and 1 targets of softmax with $\frac{\epsilon}{k-1}$, and $1-\epsilon$ respectively.

### Semi-Supervised Learning

The goal of semi-supervised learning is to use data from $P(x)$ as well as $P(x,y)$ to estimake $P(y \mid x)$. 
This combines both unsupervised and unsupervised learning. 

We can construct a model that shares parameters between a generative model $P(x)$ and a disciminiative model $P(y \mid x)$. Our cost function can be scaled between maxamizing either loss - either the MLE of the generative model or the MLE of the discriminative model. 

The generative criterion thus expresses some prior belief about the data. 

### Multitask Learning

We can 




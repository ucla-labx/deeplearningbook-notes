# Chapter 5: Machine Learning Basics

- Most machine learning algorithms have hyperparameters to set. 

- machine learning learns from experience E to do some task T with performance measure P. 

#### Machine Learning Tasks
- Enable us to solve tasks that are too difficult/too tedious to write programs for
    - such as detecting what a handwritten digit is

- Algorithms learn frm many examples; each example is composed of a collection of features that reprsent the object. 



- **classification** - function that produces categories for each input, can also deal with missing inputs 
- **regression** - predict numerical value (like housing value) given some input (like number of rooms)
- **transciprtion** - asked to transcribe some relatively unstructured representation into text - like speech recognition. 
- **machine translation** - translate a set of sequences to another set.
- **structured output** - computes a vector given some input. For example, auto-captioning images. 
- **anomaly detection** - learn a model that is capable of flagging outliers, such as credit card fraud.
- **synthesis and sampling** - generate new examples similar to teh inputs. Useful for generating textures for video games, or speech synthesis.
- **Imputation** of missing values - the algorithm is given an example $x_i$ with some of it's entries missing, and must predict the missing values. Common algorithms include low-rank matrix approximations & filtering, useful for applications like recommender systems.
- **Denoising** - the input is a corrupted example $\tilde{x}$ and the goal of the algorithm is to learn a probability distribution $p(x \mid \tilde{x})$ to predict the clean sample
- **Density estimation** - Learn a probability density function on the space from which the examples come from. To do this the model must learn about the underlying structure of the data, such as where examples tightly cluster and where they are more sparse. Can use the resulting distribution for the data imputation problem as well.

### Measuring Performance

For tasks like classification, we can measure the **accuracy** of the model - which is just the porportion of correct outputs. 
We evaluate on the **test set**, which is data we haven't trained on. 
We must design good performance metrics. 

### The Experience
- Algorithms are given access to a dataset to train on. 
- 2 types of learning algorithms: supervised and unsupervised learning

#### Supervised and Unsupervised Learning
- The dataset contains features that describe each individual example and also labels/targets. Usually involves learning/estimating a probability distribution $p(y \mid x)$ (where as unsupervised learning wants to learn $p(x)$. 
- The main difference is that in unsupervised learning, we are not provided the labels $y$. 
- Lines between different types of learning are somewhat blurred, for example if we learn a distribution $p(x,y)$ we can use that for inference by expanding it to $p(x,y) = p(y \mid x)p(x)$. 
- **Reinforcement Learning** algorithms don't experience the same dataset over and over, they interact with an environment and learn from a feedback loop/reward signal.

- dataset can be described as a **design matrix**, where each row contains a different example and each column contains a different feature. 

#### Linear Regression Example
- Given a vector $x \in R^n$ we want to predict a scalar $y$ as teh output. We define the output as a linear function of the inputs, namely $\hat{y} = w^Tx$, where $w$ are the parameters/weights of our model. 
- Next, we define a loss function: the difference between our true labels and our predicted labels. One such loss function is the MSE: $$L = \frac{1}{m} \sum_{i}(\hat{y}^i - y^{i})^{2}$$
- To minimize $MSE_{train}$, we can solve for when the gradient is 0.
  $$\nabla_w MSE_{train}= 0$$
  $$\nabla_w \frac{1}{m} \Vert \hat y^{train} - y^{train} \Vert_2^2 = 0$$
  $$\nabla_w \frac{1}{m} \Vert \hat X^{train}w - y^{train} \Vert_2^2 = 0$$
  $$w = (X^{train^T} X^{train})^{-1}X^{train^T}y^{train} $$
- we can solve these normal equations to get teh exact result.

- Generally there is also almost always a **bias term** $b$, so our true model is $\hat{y} = w^Tx + b$, and now the line doesn't have to pass through the origin. 

### Model Capacity, Overfitting, Underfitting
- Generally we train to minimize the test error but we actually caer about the gneneralization/test error, which is the expected value of how well a model will do on test inputs it has never seen before. 
- Statistical learning theory gives us some ideas on how to get a better expected testing error. First, we note that the training and test data both come from the same underlying probability distribution, often referred to as teh data generating distribution. Generally, we assume that the examples in our dataset are independent of each other and identically distributed (drawn form the same probability distribution). This distribution is called $p_{data}$ and we seek to approximate it.

#### Underfitting & Overfitting
- Model **capacity** refers to its ability to fit a wide variety of functions. For example, a polynomial model has higher capacity than a linear model. 
- When a model's capacity is too low for a particular task, they tend to underfit, when it's too high, they tend to overfit.
- If you have $n$ examples then you can perfectly fit with a model of degree $n-1$. This doesn't necessarily mean anything though. When you have more parameters than features, there are an infinte amount of functions that can git your data, and it's hard to select the one that generalizes the best. 
- While simpler functions are more likely to generalize, we need to be able to have a complex enough hypothesis in order to get lowe training error. 
- **VC dimension** is a model capacity measure - a model's VC dimeision is the largest possible $m$ such that the model can fit a dataset of $m$ different points.
- Nearest neighbor regression: when a model is asked to classify/predict, it just looks up the $k$ nearest neighbors (according to some distance metric, which itself can be learned, and returns the most common label/prediction)
- **Non parametric** models don't have fixed sized parameters. 
- **Bayes error** refers to the error an oracle who has access to the true porbability dstribution $p(x,y)$ would get. 
- **No free lunch theorem** states that no machine learning model is better than another when classifying examples over all possible data generating distributions - all models do the same when evaluated over **all possible tasks**. This means that there's no machine learning algorithm that's universally better than another one, but certain algorithms are obviously better for certain tasks. 
- 
  *
#### Reguralization
We can add preferences into our learning algorithms. For example, **weigh decay** we perfer small $w$ by adding on a term to the cost function. 
$$ J(w) = MSE_{train} + \lambda w^Tw$$
- This penalty term is called a **regularizer** and expresses preferences for one function over another. 
- Regularization is defined as adding a function to a model's loss function that's not going to reduce it's training error, but the purpose is to reduce it's expected test/generalizaiton error. 

#### Hyperparameters and Validation dataset
- Algorithms may contain several hyperparametesr (such as model capacity or $\lambda$ value for regularization) and these parameters must be set, and aren't generally learned. 
- *Don't choose hyperparameters based on their performance on the training dataset, because the learning procedure will then choose a hyperparameter setting that maximizes model capacity to fit the training dataset, which leads to overfitting.
    - For example, given a training dataset we can always fit it better by selecting a model capacity that is higher and no regularization, but this basically defeats the purpose of regularization
    - Generally we set a validation dataset aside to not train/learn on, but to validate the hyperparameters. 
    - **cross-validation** we can use cross validation when the dataset is small$-folds cross vlaidation splits the dataset into $k$ partitions and then estimates the test error across $k$ trials.

## Estimators, Bias, and Variance

- **point estimate** is any function $\hat \theta_m = g(x^1, ...,x^m)$.


- Frequentist pespective: the true paramter $\theta$ is a fixed, unknown cosntant and the point estimate is derived from a function of the data. We treat it as a random variable. 
- Function estimation and parmater estimation refer to the same thing

#### Estimator Bias
- The bias of a model/parameter is given by $bias(\hat{\theta}) = E[\hat{\theta}] - \theta$, i.e. the difference between the expected value fo the estimator and the actual value. 
- For data drawn from the Bernoulli distribution, a common practice is to estimate the mean with the sample mean, which you can show to be an unbiased estimator: 
    $$\hat{\theta} = \frac{1}{m}\sum_i x^i$$
    $$E[\frac{1}{m}\sum_i x^i] - \theta = \frac{1}{m}\sum_i E[x^i] = 0$$


#### Variance and Standard Error
- another important property of an estimator is it's **variance** or the square root - **standard error**
- Intuition for variance: if we take several datasets and repeat the learning process, what is the variance in each of our learned parameters? 
    - A high variance implies that the parameters we learn are highly dependent on the dataset, which may indicate overfitting
    - A low variance implies tht the parameters don't change much when the dataset they train on is different, which implies that there's less overfitting
- the standard error of the mean is givven by $SE(\hat \mu_m) = \frac{\sigma}{m}$
- We often compute the generalization error by takin the sample mean of the error on the test set. 

#### Bias-Variance Tradeoff
- **mean squared error** $$MSE = E[\hat \theta_m - \theta) ^2] = Bias(\hat \theta_m)^2 + Var(\hat \theta_m)$$
- relationship between bias and variance is related to underfiittin and overfitting.


#### Consistency
- Consistency refers to the idea that as the number of training examples goes to infinity, our parameter estimate approaches the true underlying parameter. 


#### Maximum Likelihood Estimation
- Essentially the goal is to find the parameter $\theta$ that maximizes our probability of observing our training data $X = x_1 ... x_m$.For example, if we have a datset of $(x, y)$ pairs and want to estimate the parameters $\theta$ that maximize our probability of observing these pairs, we can write down the likelihood invoking the iid assmputions: $L(\theta) = \prod_i p(x_i, y_i \mid \theta\\)$. 
- Another interpretation of the MLE is that it minimizes teh ference between 2 probability distributions: one which si given by your learned parameters $\theta$ and the other which is the true underlying data generating distribution. The **KL divergence** is a measure of difference between two probability distributions, and is closely related to the **cross entropy** loss. Minimizing this is equivalent to maximizing the likelihood. 


#### Conditional Log-Likelihood and MSE
The max likelihood estimator can be used to estimate a conditional probaibility $P(y, \mid x; \theta)$
$$\theta_{ML} = argmax_\theta P(Y\mid X; \theta)$$. 
- maxamizing the log-likelihood corresponds to minimizing the mean squared error. 
#### MLE properties
- Can be shown that the resulting estimate is consistent, meaning that as we have more and more training examples our parameters converge to the true values that we are trying to estimate. 
- consistent estimators can differ in their **statistical efficiency**. 
- Cramer-Rao lower bound shows that no consistent estimator has a lower MSE than MLE. 

## Bayesian Statistics
- Frequentist statistics does not account for the fact that our true parameters parametrizing our underlying probability dstribution themselves can be random  variables themselves. 
- We represent prior knowledge of our parameter $\theta$ using a prior probability distribution $p(\theta)$. 
- Next we can "update" our belief about $\theta$ once we have observed the data by computing the posterior: $p(\theta \mid x_1 ... x_m) = \frac{p(x_1 .. x_m \mid \theta) p(\theta)}{p(x)}$
- kjkjkjkjkjkj
#### Bayesian Linear Regression

- The main difference is that in addition to expressing our likelihood as $p(y \mid x, \theta) = N(y; xw, \Sigma)$ we condition our paramters $\theta$ on a normal distribution with mean $0$ and variance $\lambda$. If we write out this likelihood and expand it, then instead of getting the mean squared error we get an additional regularizing term $\alpha w^Tw$. 


### Maximum A Posteriori
While we can make use the full distribution to make predictions, sometimes this is intractable so we still need a point estimate. We can use the maximun a posteriori point estimate wihich choosed the point of max posterior probability. 
$$ \theta_{MAP} = argmax_{\theta} p(\theta \mid x) = argmax_\theta \log p(x \mid \theta) + \log p(\theta)$$
- Basically in addition to the (log) likelihood \log p(x \mid \theta)$ we have an additional prior $p(\theta)$. 

- Linear regression with prior given by $N(w; 0, \frac{1}{\lambda}I^2)$ is equavialent to linear regression with $\lambda w^Tw$ weight decay.
- ​
## Supervised Learning Algorithms
- **Logistic Regression**: predicticts $p(y = 1 \mid x; \theta) = \sigma(\theta^Tx)$. 
- **Support Vector machines** similarly rely on a linear function $w^Tx + b$, but don't output probabilities. The prediction done by taking the sign of the output function. 
- The **kernel trick** is an important contribution of the SVM. The main idea is that many ML algorithm/functions can be expressed only as dot products between examples. For example, it can be shown that the linear function $$w^Tx +b = b + \sum_{i} \alpha_i x^Tx^i$$. 
- Next, we can replace the examples $x$ with a feature transformation $\phi(x)$, which takes $x$ and possibly expands it to a very large dimensional vector. Then, we'd have to compute the feature transformations $\phi(x)$ and the inner products $\phi(x)^T\phi(x^i)$. This can turn out to be extremely expensive, or for some feature spaces, impossible. 
- This is where the **kernel** comes in. We replace the expensive inner product $\phi(x)^T\phi(x^i)$ with a function of variables in the original feature space: $k(x, x^i)$. This means that by computing this function, we can represent the dot product in a high dimensional feature space without actually computing the vector in that feature space and taking the dot product. 
- Sometimes, $k$ also enables us to represent an infinite dimensional feature space. The kernel function must satisfy certain properties, namely that it can be expressed as an inner product between 2 variables in some feature space. See [here](https://github.com/rohan-varma/CS-188/blob/master/notes/Lecture%2011%20-%20Kernel%20Methods.ipynb) for a set of notes on kernel techniques and different types of kernels.    


- The most commonly used kernel is the **gaussian kernel**  $$k(u,v) = N(u-v; 0, \sigma I)$$. It is also known as the radial basis function or RBF kernel. 
    - This performs a kind of template matching where points close to a previously classified point are most likely going to be the same class. 

- More generally there are nonparametric learning algorithms like $k$-nearest-neighbors. However knn sometimes suffers because it assumes that all features have the same relative importance. Another learning algorithm is the **deciscion tree** which divides its input space into regions. They can be trained via **entropy** or the **Gini coefficient**. 

## Unsupervised Learning Algorithms

- usuallly related to density estimation, learning to draw samples, denoising, finding a manifold, or otherwise clustering data.
- We try to find a simpker represtation of $x$ while minimzing our penalty cost. Lower-dimensional representations, sparse representations and independent representations all provide simpler representations. 

- An example of an unsupervised learning algorithm is **PCA**. The goal is to take data in a high dimensional space and project it to a lower dimensional space. Concretely, we have a design matrix $X \in R^{n * m}$ and wish to map $X$ to an $n * p$ matrix so that it's in $p$ dimensional space, where $p < m$. This is done by applying a linear transformation to $X$, namely multiplying it with a matrix that is of demision $m * p$. It turns out that this matrix is composed of the top $p$ eigenvectors of $X^TX$. 
- FInds the directions of maximal variance, and those are the orthogonal eigenvectors that compose teh transformation matrix. 

- PCA learns an orthogonal, linear transformation of the data. It maps data such that each element is uncorrelated. 

#### $k$-means Clustering
- $k$-meanns clustering divides the training set into $k$ different clusters. 
- It initialized $k$ different centroids and in each step, each training example is first assigned to the nearest cluster. After that, each centroid is updated to be the mean of all of the examples in the cluster. 
- It's difficult to know if our clusters are actually meaningful. Moreover, we may want to use a distributed representation rather than a one-hot representation to capture more than one attribute per example.

## Stochastic Gradient Descent 

- Essentially an extension of Gradient Descent
- Main idea is that computing the true gradient may be inefficient or impractical, especially on a very large deep learning dataset, since we need to sum across all of the training examples, and there could be millions. 
- Instead, stochastic GD samples a **minibatch** of training examples, computes the gradients using only those examples, and updates the weight according to that gradient.
    - When the size of the minibatch = 1, this is referred to as **online learning**
    - Provides a noisy estimate of the true gradient. 
    - Provides a scalable way of training nonlinear models on large datasets. Prior to this the main approach was to use the kernel trick, which required an $O(m^2)$ kernel matrix. 
    - Main insight is to treat the gradient as an expectation
    - ​

$$ g = \frac{1}{m'} \nabla_\theta \sum_{i=1}^{m'}L(x_i, y_i, \theta) $$


SGD is $O(k)$ runtime instead of $O(m)$ where $k$ is the batch size, and is $k << m$.

#### Constructing a Machine Learning Algorithm
- Define a loss function $J$ of your model predictions and the true labels
    - Mean squared error, cross entropy, and hinge loss are examples
    - Define a function that gives the probability distribtion $p(y \mid x)$, following for example MLE or MAP estimation
    - Add a regularizer $\lambda R(w)$ that penalizes complex models. 
    - Learn parameters with SGD. 

## Challenges Motivating Deep Learning

- **curse of dimensionality** - when we have a high n dimesions, the numCCCber of possible values increases exponentially. 

- Curse of dimensionality example: say we have a 10-dimensional feature vector, where each feature can range between 100 different values. That means that the data space we are working in is $10^100$ dimensional, and there's no way we can get enough data samples to even get a small fraction of that space. 
- Basically the idea of the curse of dimensionality is that we are generally working with a small cluster of data from the true data space. As the dimensionaility increases, points grow further and further apart as the space becomes more sparse and clustered. 
- There are certain priors about the functions that we want to learn. In particular, we want to learn a finction $$f*(x) \approx f*(x+\epsilon)$$. That is, the output of our function on some unseen datapoint that is close to a datapoint is closely related to the output of that close seen datapoint.

- A local kernel used by a kernel machine can be thought of as a similarlity functiont hat performs a "template matching" by measuring how much an example $x$ is similar to a training example of $x_i$. 
- Some of the modern motivation for deep learning is due to overcoming the limits of this template matching approach.

- The main question is that can we represent a complicated function, such as one that distinguishes between $O(2^k)$ regions, without having an exponential number of training examples? 
- This is difficult for many traditional machine learning algorithms, but the key idea is that deep learning can do this well by introducing dependencies between regions and making adssumptions about the data generating distribution. 
- Core idea of deep learning: assume tht the data were generated by a composition of the data's features, in a hierarchical fashion.
    - Justification of the hierarchical fashion of neural network layers, each layer learns a higher and higher level representation that is based on a (combination of) the features in the previous layer.

#### Manifold Learning
- **manifold** is a lower dimesional connected region. Each dimension corresponds to a local direction of variation. 

- **Manifold learning** algorithms assume that the input space $R^n$ consists of mostly invalid inputs (i.e. feature vectors that cannot actually occur), and that inputs occur only on a collection of different manifolds in the space $R^n$. 
- Variations in the outputs occur only along directions that lie on the manifolds, or when the input moves from one manifold to another. 
- So basically the output is assumed to only come from manifolds of the data, not the whole data space. 

- **manifold hypothesis** is supported by the fact that text,speech, and image data is all fairly structured and doesn't resemble unioform noise. We can also think of these directions of variation as transformations. For example, rotating an object may move a training example along it's manifold. 
  The data is assumed to come from a concentrated probability distributions, that can be represented by manifolds. For example, in image recognition, you're not likely to have an input that consists of uniform random noise. 

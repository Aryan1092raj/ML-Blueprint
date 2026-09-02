# Logistic Regression — Derivation

## 1. The Model

For a dataset with $n$ features, the logistic regression model predicts the probability of the positive class as:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

where the linear predictor $z$ is:

$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

In vectorized notation:

$$z = Xw + b$$

$$\hat{y} = \sigma(Xw + b)$$

Where:
- **$X$** is the input feature matrix of shape $(N, n)$
- **$w$** is the weight vector of shape $(n, 1)$
- **$b$** is the bias term (scalar)
- **$N$** is the number of training examples
- **$\sigma(\cdot)$** is the sigmoid function
- **$\hat{y}$** is the predicted probability of class 1

## 2. The Loss Function

We use **log-loss** (binary cross-entropy) as our loss function:

$$L(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Why log-loss?**
- It comes from Maximum Likelihood Estimation assuming Bernoulli-distributed labels.
- It is convex with respect to the parameters (unlike MSE applied to sigmoid outputs).
- It heavily penalizes confident wrong predictions (e.g., predicting 0.99 when true label is 0).
- The gradient has a simple, numerically stable form.

## 3. Computing the Gradients

To minimize the loss, we compute the gradient of $L$ with respect to $w$ and $b$. The gradient points in the direction of steepest increase, so we move in the opposite direction.

### 3.1 Derivative of the sigmoid

First, a useful identity for the sigmoid derivative:

$$\sigma'(z) = \frac{d}{dz} \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$

### 3.2 Derivative of loss with respect to $z$

For a single training example $i$, the loss is:

$$\ell_i = - \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Derivative with respect to $\hat{y}^{(i)}$:

$$\frac{\partial \ell_i}{\partial \hat{y}^{(i)}} = -\frac{y^{(i)}}{\hat{y}^{(i)}} + \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} = \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})}$$

Now apply chain rule through the sigmoid:

$$\frac{\partial \ell_i}{\partial z^{(i)}} = \frac{\partial \ell_i}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}}$$

$$= \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})} \cdot \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

$$= \hat{y}^{(i)} - y^{(i)}$$

**Beautiful simplification:** the messy sigmoid terms cancel out, leaving just the prediction error.

### 3.3 Derivative with respect to Weights ($w$)

Using the chain rule again:

$$\frac{\partial \ell_i}{\partial w_j} = \frac{\partial \ell_i}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j} = (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

Averaging over all $N$ training examples:

$$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

In **vectorized form** (all weights at once):

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)$$

### 3.4 Derivative with respect to Bias ($b$)

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})$$

## 4. Gradient Descent Update Rule

We update the parameters iteratively, moving in the **opposite** direction of the gradient:

$$w := w - \alpha \frac{\partial L}{\partial w}$$

$$b := b - \alpha \frac{\partial L}{\partial b}$$

Where $\alpha$ is the learning rate.

Substituting the gradients:

$$w := w - \frac{\alpha}{N} X^T (\hat{y} - y)$$

$$b := b - \frac{\alpha}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})$$

We repeat until convergence (loss stops decreasing significantly).

## 5. Prediction

For a new input $X$, the predicted probability is:

$$\hat{y} = \sigma(Xw + b)$$

The class prediction uses a threshold (typically 0.5):

$$\text{class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$
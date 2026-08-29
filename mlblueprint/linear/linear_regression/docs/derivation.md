# Linear Regression — Derivation

## 1. The Model

For a dataset with $n$ features, the linear regression model predicts the output $\hat{y}$ as:

$$\hat{y} = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

In vectorized notation:

$$\hat{y} = Xw + b$$

Where:
- **$X$** is the input feature matrix of shape $(N, n)$
- **$w$** is the weight vector of shape $(n, 1)$
- **$b$** is the bias term (scalar)
- **$N$** is the number of training examples

## 2. The Loss Function

We use the **Mean Squared Error (MSE)** as our loss function:

$$L(w, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2$$

**Why MSE?**
- Squaring penalizes larger errors more heavily than smaller ones.
- MSE is differentiable everywhere, which allows us to use gradient-based optimization.
- Under the assumption that errors are normally distributed, minimizing MSE is equivalent to Maximum Likelihood Estimation.

*(Note: The division by $2$ is a mathematical convenience. When we take the derivative, the exponent $2$ comes down and cancels it out, making the expressions cleaner.)*

## 3. Computing the Gradients

To minimize the loss, we compute the gradient of $L$ with respect to each parameter. The gradient points in the direction of steepest increase, so we will later move in the opposite direction.

### 3.1 Derivative with respect to Weights ($w$)

We apply the **chain rule**. For a single weight $w_j$ and a single training example $i$:

$$\frac{\partial \ell_i}{\partial w_j} = \frac{\partial \ell_i}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial w_j}$$

**Step 1:** Derivative of loss with respect to prediction:

$$\frac{\partial \ell_i}{\partial \hat{y}^{(i)}} = (\hat{y}^{(i)} - y^{(i)})$$

*(The $\frac{1}{2}$ and the exponent $2$ cancel out.)*

**Step 2:** Derivative of prediction with respect to weight:

Since $\hat{y}^{(i)} = \sum_{k} w_k x_k^{(i)} + b$, the derivative with respect to $w_j$ is simply:

$$\frac{\partial \hat{y}^{(i)}}{\partial w_j} = x_j^{(i)}$$

**Step 3:** Combine using chain rule:

$$\frac{\partial \ell_i}{\partial w_j} = (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

**Step 4:** Average over all $N$ training examples:

$$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

In **vectorized form** (all weights at once):

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)$$

### 3.2 Derivative with respect to Bias ($b$)

Same chain rule approach:

$$\frac{\partial \ell_i}{\partial b} = \frac{\partial \ell_i}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial b}$$

Since $\frac{\partial \hat{y}^{(i)}}{\partial b} = 1$:

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})$$

## 4. Gradient Descent Update Rule

We update the parameters iteratively, moving in the **opposite** direction of the gradient (steepest descent) to minimize the loss. The learning rate $\alpha$ controls the step size:

$$w := w - \alpha \frac{\partial L}{\partial w}$$

$$b := b - \alpha \frac{\partial L}{\partial b}$$

We repeat this until convergence (i.e., the loss stops decreasing significantly).

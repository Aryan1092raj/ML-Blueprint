# Logistic Regression — Intuition

## The Big Idea

Imagine you're a bank deciding whether to approve a loan application. You have the applicant's income and credit score, and you know from past data whether similar applicants paid back or defaulted.

| Income ($k) | Credit Score | Repaid? |
|---|---|---|
| 80 | 750 | Yes |
| 35 | 580 | No |
| 120 | 800 | Yes |
| 25 | 520 | No |

Linear regression would try to draw a straight line through this, but that doesn't make sense — the answer isn't a number, it's **yes or no**. What you really want is a **probability**: "Given this income and credit score, what's the chance they'll repay?"

**That's what Logistic Regression does.** It takes a linear combination of your features and squashes it through an S-shaped curve (the sigmoid) to give you a probability between 0 and 1.

## What is it actually doing?

Logistic regression starts with the same linear model:
$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

But instead of using `z` directly, it passes it through the **sigmoid function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This squashes any number into the range (0, 1), which we interpret as the probability of the positive class.

## How does it learn?

We still use gradient descent, but the loss function changes. We can't use MSE anymore — it's not convex for logistic regression. Instead we use **log-loss** (cross-entropy):

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where $\hat{y}^{(i)} = \sigma(z^{(i)})$ is the predicted probability.

The gradient has a beautifully simple form:
$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)$$

This is almost the same as linear regression — just replace the raw prediction with the probability.

## When would you use it?

- Predicting whether an email is spam (yes/no)
- Medical diagnosis: disease present or absent based on test results
- Credit approval: will this person default?
- Any binary classification problem where you want probabilities, not just hard labels

## When would you NOT use it?

- When classes aren't linearly separable (the decision boundary is curved)
- When you have more than two classes without modification (need multinomial/softmax)
- When you need complex feature interactions (trees/neural nets handle this better)

## The key takeaway

Logistic regression is linear regression that went to probability school. It keeps the simplicity and interpretability of a linear model but outputs calibrated probabilities. The only real trick is keeping the log-loss numerically stable when probabilities get close to 0 or 1.
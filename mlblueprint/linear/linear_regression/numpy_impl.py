import numpy as np

class LinearRegression:
    
    """
    Ridge Regression (L2 Regularized Linear Regression) using Gradient Descent.
    
    Loss = (1/2N) * sum((y_hat - y)^2) + lam * sum(w^2)
    
    Parameters
    ----------
    lr : float
        Learning rate. Default 0.01.
    n_iters : int
        Number of gradient descent iterations. Default 1000.
    lam : float
        L2 regularization strength. Set to 0 for plain Linear Regression.
    """

    def __init__(self, lr=0.01, n_iters=1000, lam=0.1):
        self.lr = lr
        self.n_iters = n_iters 
        self.lam = lam 
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.random.random(n_features)
        self.bias = np.random.random()

        # training loop
        for epoch in range(self.n_iters):

            # forward pass 
            y_hat = X@self.weights + self.bias

            # loss
            mse = np.mean((y_hat-y)**2)/2 + self.lam*np.sum((self.weights**2))

            # print loss
            if epoch%100==0:
                print(f"epoch {epoch} loss: {mse}")

            # backpropagation
            dw = (1/n_samples)*(np.dot(X.T,y_hat-y)) + 2*self.lam*self.weights
            db = (1/n_samples)*(np.sum(y_hat-y))

            # update 
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        X = np.array(X)
        y_hat = X@self.weights + self.bias
        return y_hat


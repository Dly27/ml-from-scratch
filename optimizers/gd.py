import numpy as np
from functions import *

class Optimizer:
    def step(self, params, *args, **kwargs):
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, loss, lr):
        self.lr = lr
        self.loss = loss
        self.gradient_function = gradients[loss]

    def step(self, params, X, y):
        grad = self.gradient_function(params, X, y)
        return params - self.lr * grad

class SGD(Optimizer):
    def __init__(self, loss, lr):
        self.lr = lr
        self.loss = loss
        self.gradient_function = gradients[loss]

    def step(self, params, X, y):
        i = np.random.randint(X.shape[0])
        grad = self.gradient_function(params, X[i:i+1], y[i:i+1])
        return params - self.lr * grad

class MBGD(Optimizer):
    def __init__(self, loss, lr, batch_size):
        self.lr = lr
        self.loss = loss
        self.gradient_function = gradients[loss]
        self.batch_size = batch_size

    def step(self, params, X, y):
        indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
        grad = self.gradient_function(params, X[indices], y[indices])
        return params - self.lr * grad

class Momentum(Optimizer):
    def __init__(self, loss, lr, momentum):
        self.lr = lr
        self.loss = loss
        self.gradient_function = gradients[loss]
        self.momentum = momentum
        self.velocity = None

    def step(self, params, X, y):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        i = np.random.randint(X.shape[0])
        grad = self.gradient_function(params, X[i:i+1], y[i:i+1])
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return params - self.velocity

class Adam(Optimizer):
    def __init__(self, loss, lr, b1=0.9, b2=0.999):
        self.lr = lr
        self.loss = loss
        self.gradient_function = gradients[loss]
        self.momentum = None
        self.velocity = None
        self.t = 0
        self.b1 = b1
        self.b2= b2
        self.momentum_corrected = None
        self.velocity_corrected = None
        self.epsilon = 10 ** -8

    def step(self, params, X, y):
        self.t += 1

        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        if self.momentum is None:
            self.momentum = np.zeros_like(params)

        grad = self.gradient_function(params, X, y)
        self.velocity = self.b2 * self.velocity + (1 - self.b2) * (grad ** 2)
        self.momentum = self.b1 * self.momentum + (1 - self.b1) * grad
        self.momentum_corrected = self.momentum / (1 - self.b1 ** self.t)
        self.velocity_corrected = self.velocity / (1 - self.b2 ** self.t)

        return params - self.lr * (self.momentum_corrected / (np.sqrt(self.velocity_corrected) + self.epsilon) )

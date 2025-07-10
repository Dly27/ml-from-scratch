import numpy as np

def quadratic(x, a):
    return np.sum((x-a) ** 2)

def grad_quadratic(x,a):
    return 2 * (x-a)

def rosenbrock_vec(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock_vec(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def mse_loss(X, y, w):
    y_pred = X @ w
    return np.mean((y - y_pred)**2)

def grad_mse_loss(X, y, w):
    y_pred = X @ w
    return (-2 / X.shape[0]) * X.T @ (y - y_pred)

def mae_loss(X, y, w):
    y_pred = X @ w
    return np.mean(np.abs(y - y_pred))

def grad_mae_loss(X, y, w):
    y_pred = X @ w
    sign = np.sign(y_pred - y)
    return (1 / X.shape[0]) * X.T @ sign

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(X, y, w):
    z = X @ w
    y_pred = sigmoid(z)
    epsilon = 1e-8
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

def grad_logistic_loss(X, y, w):
    z = X @ w
    y_pred = sigmoid(z)
    return (1 / X.shape[0]) * X.T @ (y_pred - y)

def huber_loss(X, y, w, delta=1.0):
    y_pred = X @ w
    error = y - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

def grad_huber_loss(X, y, w, delta=1.0):
    y_pred = X @ w
    error = y - y_pred
    mask = np.abs(error) <= delta
    grad = np.where(mask, -error, -delta * np.sign(error))
    return (1 / X.shape[0]) * X.T @ grad

gradients = {
    "quadratic": {"func": quadratic, "grad": grad_quadratic},
    "rosenbrock_vec": {"func": rosenbrock_vec, "grad": grad_rosenbrock_vec},
    "mse_loss": {"func": mse_loss, "grad": grad_mse_loss},
    "mae_loss": {"func": mae_loss, "grad": grad_mae_loss},
    "logistic_loss": {"func": logistic_loss, "grad": grad_logistic_loss},
    "huber_loss": {"func": huber_loss, "grad": grad_huber_loss},
}
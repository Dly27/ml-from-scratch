import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
    def _prep_features(self, x):
        x = np.array(x)

        if np.ndim(x) == 1:
            x = x.reshape(-1,1)

        x = x / np.mean(x)

        return np.hstack((np.ones((x.shape[0], 1)), x))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, x, y):
        x = self._prep_features(x)
        y = np.array(y)
        predictions = self.sigmoid(x @ self.weights)

        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def fit(self, x, y, lr, iterations):
        x = self._prep_features(x)
        y = np.array(y)
        self.weights = np.zeros(x.shape[1])

        for i in range(iterations):
            predictions = self.sigmoid(x @ self.weights)
            error = predictions - y
            gradient = (1/x.shape[0]) * (x.T @ error)
            self.weights += lr * gradient

            if i % 10 == 0:
                current_loss = self.loss(x, y)
                print(f"Iteration {i}, Loss: {current_loss:.4f}")

    def predict(self, x):
        x = self._prep_features(x)
        predictions = self.sigmoid(x @ self.weights)
        return (predictions >= 0.5).astype(int)



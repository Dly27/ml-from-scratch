import numpy as np
class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def _prepare_features(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def fit(self, x, y):
        y = np.array(y)
        x = self._prepare_features(x)

        self.coefficients = np.linalg.pinv(x) @ y

        self.x = x
        self.y = y

        return self.coefficients

    def predict(self, x_new):
        x_new = self._prepare_features(x_new)

        print(x_new @ self.coefficients)

    def score(self, x_test, y_test):
        y_test = np.array(y_test)
        x_test = self._prepare_features(x_test)

        y_pred = x_test @ self.coefficients

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        return 1 - ss_res / ss_tot




import numpy as np

class KNN:
    def __init__(self):
        pass

    def _prep_features(self, x):
        return np.array(x)

    def fit(self, x, y):
        self.x = self._prep_features(x)
        self.y = y

    def majority_vote(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

    def predict(self, x_test, k):
        x_test = self._prep_features(x_test)
        predictions = []

        for x in x_test:
            distances = np.sqrt(np.sum((self.x - x)**2, axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = self.y[nearest_indices]
            predictions.append(self.majority_vote(nearest_labels))

        return predictions

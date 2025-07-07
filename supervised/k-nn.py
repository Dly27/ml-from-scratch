import numpy as np
import heapq

import sklearn.datasets
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import functools
import time

class BallTreeNode:
    def __init__(self, center, radius, points=None, labels=None):
        self.center = center
        self.radius = radius
        self.points = points
        self.labels = labels
        self.left = None
        self.right = None

class KNN:
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.root = None
        self.algorithm = "brute"

    def _prep_features(self, x):
        return np.array(x)

    def fit(self, x, y, algorithm="brute"):
        self.x = self._prep_features(x)
        self.y = np.array(y)
        self.algorithm = algorithm

        if algorithm == "ball_tree":
            self.root = self.build_ball_tree(self.x, self.y)
        elif algorithm == "ball_star_tree":
            self.root = self.build_ball_star_tree(self.x, self.y)

    def PCA(self, points):
        center = np.mean(points, axis=0)
        centered_points = points - center
        u, s, vh = np.linalg.svd(centered_points, full_matrices=False)
        return centered_points @ vh[0], center

    def majority_vote(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

    def safe_tree_recursion(func):
        @functools.wraps(func)
        def wrapper(self, points, labels, *args, **kwargs):
            if points is None or len(points) == 0:
                return None
            return func(self, points, labels, *args, **kwargs)
        return wrapper

    def predict_brute(self, x_test, k):
        x_test = self._prep_features(x_test)
        predictions = []

        for x in x_test:
            distances = np.sqrt(np.sum((self.x - x)**2, axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = self.y[nearest_indices]
            predictions.append(self.majority_vote(nearest_labels))

        return predictions

    @safe_tree_recursion
    def build_ball_tree(self, points, labels):
        n = len(points)
        if n <= self.leaf_size:
            center = np.mean(points, axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1)) if n > 0 else 0
            return BallTreeNode(center, radius, points, labels)

        idx1 = np.random.randint(0, n)
        point1 = points[idx1]
        distances = np.linalg.norm(points - point1, axis=1)
        idx2 = np.argmax(distances)
        point2 = points[idx2]
        left_points, left_labels = [], []
        right_points, right_labels = [], []

        for i in range(n):
            d1 = np.linalg.norm(points[i] - point1)
            d2 = np.linalg.norm(points[i] - point2)

            if d1 < d2:
                left_points.append(points[i])
                left_labels.append(labels[i])
            else:
                right_points.append(points[i])
                right_labels.append(labels[i])

        # Check for empty splits to avoid infinite recursion
        if len(left_points) == 0 or len(right_points) == 0:
            center = np.mean(points, axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1))
            return BallTreeNode(center, radius, points, labels)

        left = self.build_ball_tree(np.array(left_points), np.array(left_labels))
        right = self.build_ball_tree(np.array(right_points), np.array(right_labels))

        center = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))

        node = BallTreeNode(center, radius)
        node.left = left
        node.right = right

        return node

    @safe_tree_recursion
    def build_ball_star_tree(self, points, labels):
        if len(points) <= self.leaf_size:
            center = np.mean(points, axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1)) if len(points) > 0 else 0
            return BallTreeNode(center, radius, points, labels)

        # Find first principle component projections
        projections, center = self.PCA(points)
        sorted_indices = np.argsort(projections)
        points = points[sorted_indices]
        labels = labels[sorted_indices]

        # Find best split
        best_score = float("inf")
        n = len(points)
        num_candidates = min(n, 10)
        step = max(1, n // num_candidates)

        for i in range(step, n - step, step):
            left_points = points[:i]
            right_points = points[i:]
            left_center = np.mean(left_points, axis=0)
            right_center = np.mean(right_points, axis=0)
            left_radius = np.max(np.linalg.norm(left_points - left_center, axis=1))
            right_radius = np.max(np.linalg.norm(right_points - right_center, axis=1))

            balance_term = abs(len(left_points) - len(right_points)) / n
            overlap = max(0, (left_radius + right_radius - np.linalg.norm(left_center - right_center)))
            score = 0.8 * overlap + 0.2 * balance_term

            if score < best_score:
                best_score = score
                best_idx = i

        # Check for no good split or empty splits
        if best_idx is None or best_idx == 0 or best_idx == n:
            center = np.mean(points, axis=0)
            radius = np.max(np.linalg.norm(points - center, axis=1))
            return BallTreeNode(center, radius, points, labels)

        left = self.build_ball_star_tree(points[:best_idx], labels[:best_idx])
        right = self.build_ball_star_tree(points[best_idx:], labels[best_idx:])

        center = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))

        node = BallTreeNode(center, radius)
        node.left = left
        node.right = right
        return node

    def ball_tree_search(self, node, target, k, heap):
        if node is None:
            return

        if node.points is not None:
            for i in range(len(node.points)):
                distance = np.linalg.norm(target - node.points[i])
                if len(heap) < k:
                    heapq.heappush(heap, (-distance, node.labels[i]))
                else:
                    if distance < -heap[0][0]:
                        heapq.heappushpop(heap, (-distance, node.labels[i]))
            return

        tau = -heap[0][0] if len(heap) == k else float('inf')

        distance_left = np.linalg.norm(target - node.left.center) if node.left else float('inf')
        distance_right = np.linalg.norm(target - node.right.center) if node.right else float('inf')

        # Search closer child first
        if distance_left < distance_right:
            if distance_left - node.left.radius <= tau:
                self.ball_tree_search(node.left, target, k, heap)
                tau = -heap[0][0] if len(heap) == k else float('inf')
            if distance_right - node.right.radius <= tau:
                self.ball_tree_search(node.right, target, k, heap)
        else:
            if distance_right - node.right.radius <= tau:
                self.ball_tree_search(node.right, target, k, heap)
                tau = -heap[0][0] if len(heap) == k else float('inf')
            if distance_left - node.left.radius <= tau:
                self.ball_tree_search(node.left, target, k, heap)

    def predict(self, x_test, k):
        x_test = self._prep_features(x_test)
        predictions = []

        if self.algorithm == "brute":
            return self.predict_brute(x_test, k)

        elif self.algorithm == "ball_tree" or self.algorithm == "ball_star_tree":
            for xi in x_test:
                heap = []
                self.ball_tree_search(self.root, xi, k, heap)
                k_labels = [label for _, label in sorted(heap, key=lambda xi: -xi[0])]
                predictions.append(self.majority_vote(k_labels))
            return predictions

def accuracy_score(target, predicted):
    return np.mean(np.array(target) == np.array(predicted))


def benchmark(model, X, y, k, algorithm):
    start_fit = time.time()
    model.fit(X, y, algorithm=algorithm)
    fit_duration = time.time() - start_fit

    start_pred = time.time()
    predictions = model.predict(X, k)
    pred_duration = time.time() - start_pred

    acc = accuracy_score(y, predictions)
    return acc, fit_duration, pred_duration

if __name__ == "__main__":
    np.random.seed(42)

    X, y = make_classification(n_samples=1000, n_features=50)

    model1 = KNN(leaf_size=2)
    model2 = KNN(leaf_size=2)

    acc1, fit_time1, pred_time1 = benchmark(model1, X, y, 3, algorithm="ball_tree")
    acc2, fit_time2, pred_time2 = benchmark(model2, X, y, 3, algorithm="ball_star_tree")

    print("Ball Tree Accuracy:", acc1)
    print("Ball Tree Fit Time: ", fit_time1)
    print("Ball Tree Predict Time:", pred_time1)

    print("Ball* Tree Accuracy:", acc2)
    print("Ball* Tree Fit Time: ", fit_time2)
    print("Ball* Tree Predict Time:", pred_time2)

    print("Percentage difference in fit time: ", (fit_time1 - fit_time2) * 100 / fit_time1)
    print("Percentage difference in predict time: ", (pred_time1 - pred_time2) * 100 / pred_time1)

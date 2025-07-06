import numpy as np
import heapq

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
            self.root = self.build_tree(self.x, self.y)

    def majority_vote(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

    def predict_brute(self, x_test, k):
        x_test = self._prep_features(x_test)
        predictions = []

        for x in x_test:
            distances = np.sqrt(np.sum((self.x - x)**2, axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = self.y[nearest_indices]
            predictions.append(self.majority_vote(nearest_labels))

        return predictions

    def build_tree(self, points, labels):
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

        left = self.build_tree(np.array(left_points), np.array(left_labels))
        right = self.build_tree(np.array(right_points), np.array(right_labels))

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
                dist = np.linalg.norm(target - node.points[i])
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, node.labels[i]))
                else:
                    if -dist > heap[0][0]:
                        heapq.heappushpop(heap, (-dist, node.labels[i]))
            return

        distance_to_center = np.linalg.norm(target - node.center)

        if node.left and distance_to_center < node.left.radius:
            self.ball_tree_search(node.left, target, k, heap)
            if node.right and distance_to_center - node.left.radius <= node.right.radius:
                self.ball_tree_search(node.right, target, k, heap)
        else:
            self.ball_tree_search(node.right, target, k, heap)
            if node.left and distance_to_center - node.right.radius <= node.left.radius:
                self.ball_tree_search(node.left, target, k, heap)

    def predict(self, x_test, k):
        x_test = self._prep_features(x_test)
        predictions = []

        if self.algorithm == "brute":
            return self.predict_brute(x_test, k)

        elif self.algorithm == "ball_tree":
            for x in x_test:
                heap = []
                self.ball_tree_search(self.root, x, k, heap)
                k_labels = [label for _, label in sorted(heap, key=lambda x: -x[0])]
                predictions.append(self.majority_vote(k_labels))
            return predictions

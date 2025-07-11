# https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf

import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, states):
        self.states = states
        self.parent = None

class RRT:
    def __init__(self, x_init):
        self.x_init = np.array(x_init)
        self.nodes = [Node(self.x_init)]

    def add_state(self, states, parent):
        node = Node(states)
        node.parent = parent
        self.nodes.append(node)

    def nearest_neighbour(self, x_random):
        nearest = self.nodes[0]
        min_distance = np.linalg.norm(x_random - nearest.states)

        for node in self.nodes:
            distance = np.linalg.norm(x_random - node.states)

            if distance < min_distance:
                nearest = node
                min_distance = distance

        return nearest

    def select_control(self, x_near, x_random, step):
        direction = x_random - x_near.states
        norm = np.linalg.norm(direction)

        if norm == 0:
            return np.zeros_like(direction)

        return (direction / norm) * step

    def new_state(self, x_near, u, step):
        return x_near.states + u * step
    def valid(self, x):
        return True

    def grow(self, bounds, k, goal, r):
        bounds = np.array(bounds)
        for i in range(k):
            x_random = np.array([np.random.uniform(low, high) for low, high in bounds])
            x_near = self.nearest_neighbour(x_random)
            u = self.select_control(x_near, x_random, step=0.5)
            x_new = self.new_state(x_near, u, step=0.5)

            if self.valid(x_new):
                self.add_state(x_new, x_near)

            if np.linalg.norm(x_new - goal) < r:
                break

    def get_path(self, goal_node):
        path = []
        current = goal_node
        while current is not None:
            path.append(current.states)
            current = current.parent
        return path[::-1]

# ========== RUNNING TEST ==========

bounds = [(0, 10), (0, 10)]
start = [0, 0]
goal_coords = np.array([9, 9])
rrt = RRT(x_init=start)
rrt.grow(bounds=bounds, k=1000, goal=goal_coords, r=0.1)
nearest_to_goal = rrt.nearest_neighbour(goal_coords)
path = rrt.get_path(nearest_to_goal)

# Plotting
xs = [node.states[0] for node in rrt.nodes]
ys = [node.states[1] for node in rrt.nodes]
plt.scatter(xs, ys, s=5, color='gray', label='Tree Nodes')

# Draw tree edges
for node in rrt.nodes:
    if node.parent is not None:
        x1, y1 = node.states
        x2, y2 = node.parent.states
        plt.plot([x1, x2], [y1, y2], 'lightblue', linewidth=0.5)

# Draw path
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal_coords, color='blue', label='Goal')
plt.legend()
plt.title("RRT Path Planning")
plt.axis('equal')
plt.show()
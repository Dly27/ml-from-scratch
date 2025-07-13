# Paper: https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time


def bresenham(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class Node:
    def __init__(self, states):
        self.states = states
        self.parent = None


class RRT:
    def __init__(self, x_init, grid_map, rebuild_freq, goal, goal_bias):
        self.x_init = np.array(x_init)
        self.nodes = [Node(self.x_init)]
        self.grid_map = np.array(grid_map)
        self.map_height, self.map_width = self.grid_map.shape
        self.node_positions = [self.x_init]
        self.kd_tree = cKDTree(np.array(self.node_positions))
        self.kd_tree_needs_update = False
        self.rebuild_freq = rebuild_freq
        self.path = []
        self.smoothed_path = []
        self.goal_bias = goal_bias
        self.goal = goal

    def add_state(self, states, parent):
        node = Node(states)
        node.parent = parent
        self.nodes.append(node)
        self.node_positions.append(states)

        # Rebuild kd-tree only every rebuild_freq
        if len(self.nodes) % self.rebuild_freq == 0:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False
        else:
            self.kd_tree_needs_update = True

    def find_nearest_neighbour(self, x_random):
        if self.kd_tree_needs_update:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False
        _, idx = self.kd_tree.query(x_random)
        return self.nodes[idx]

    def select_control(self, x_near, x_random, step):
        direction = x_random - x_near.states
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros_like(direction)
        return (direction / norm) * step

    def new_state(self, x_near, u, step):
        return x_near.states + u * step

    def valid(self, x1, x2=None):
        # Check nodes are in valid positions (In bounds / not in obstacles)
        if x2 is None:
            x = np.round(x1).astype(int)
            x_idx, y_idx = x[0], x[1]
            if (0 <= x_idx < self.map_width) and (0 <= y_idx < self.map_height):
                return self.grid_map[y_idx, x_idx] == 0
            return False
        # Check edges are in valid positions
        else:
            x0, y0 = np.floor(x1).astype(int)
            x1_, y1_ = np.floor(x2).astype(int)
            for x, y in bresenham(x0, y0, x1_, y1_):
                if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                    return False
                if self.grid_map[y, x] == 1:
                    return False
            return True

    def grow(self, k, r):
        # Use algorithm from the paper
        for _ in range(k):
            if np.random.rand() < self.goal_bias:
                x_random = self.goal
            x_random = np.array([
                np.random.uniform(0, self.map_width),
                np.random.uniform(0, self.map_height)
            ])
            x_near = self.find_nearest_neighbour(x_random)
            u = self.select_control(x_near, x_random, step=1)
            x_new = self.new_state(x_near, u, step=1)

            if self.valid(x_near.states, x_new):
                self.add_state(x_new, x_near)

            if np.linalg.norm(x_new - self.goal) < r:
                break

        # Ensure kd tree rebuilt at very end
        if self.kd_tree_needs_update:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False

    def get_path(self, goal_node):
        current = goal_node
        while current is not None:
            self.path.append(current.states)
            current = current.parent
        self.path = self.path[::-1]
        return self.path

    def smooth_path(self):
        smooth = [self.path[0]]
        i = 0
        while i < len(self.path) - 1:
            j = len(self.path) - 1
            while j > i:
                if self.valid(self.path[j], self.path[i]):
                    smooth.append(self.path[j])
                    i = j
                    break
                j -= 1
        self.smoothed_path = smooth
        return self.smoothed_path

    def path_cost(self):
        return sum(np.linalg.norm(self.path[i + 1] - self.path[i]) for i in range(len(self.path) - 1))


# ========== RUNNING TEST ==========

grid_map = np.zeros((100, 100), dtype=int)

start_coords = [50, 80]
goal_coords = np.array([90, 90])

for i in range(99):
    for j in range(99):
        if i % 2 == 0 and j % 2 == 0:
            grid_map[i][j] = 1

grid_map[80][50] = 0

# Create RRT
start = time.time()
rrt = RRT(x_init=start_coords, grid_map=grid_map, rebuild_freq=50, goal=goal_coords, goal_bias=0.05)
rrt.grow(k=5000, r=0.5)
end = time.time()
print("RUn time: ", end - start)

# Find path from start to goal
nearest_to_goal = rrt.find_nearest_neighbour(goal_coords)
path = rrt.get_path(nearest_to_goal)
smooth_path = rrt.smooth_path()
print(rrt.path_cost())

# Draw nodes
xs = [node.states[0] for node in rrt.nodes]
ys = [node.states[1] for node in rrt.nodes]
plt.scatter(xs, ys, s=5, color='gray', label='Tree Nodes')

# Draw edges
for node in rrt.nodes:
    if node.parent is not None:
        x1, y1 = node.states
        x2, y2 = node.parent.states
        plt.plot([x1, x2], [y1, y2], 'lightblue', linewidth=0.5)

# Draw obstacles
for y in range(grid_map.shape[0]):
    for x in range(grid_map.shape[1]):
        if grid_map[y, x] == 1:
            plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

# PLot
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')
plt.scatter(*start_coords, color='green', label='Start')
plt.scatter(*goal_coords, color='blue', label='Goal')
plt.legend()
plt.title("RRT with Grid Map")
plt.axis('equal')
plt.grid(True)
plt.show()

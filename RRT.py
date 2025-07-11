import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class Node:
    def __init__(self, states):
        self.states = states
        self.parent = None

class RRT:
    def __init__(self, x_init, grid_map, rebuild_freq=50):
        self.x_init = np.array(x_init)
        self.nodes = [Node(self.x_init)]
        self.grid_map = np.array(grid_map)
        self.map_height, self.map_width = self.grid_map.shape
        self.node_positions = [self.x_init]
        self.kd_tree = cKDTree(np.array(self.node_positions))
        self.kd_tree_needs_update = False
        self.rebuild_freq = rebuild_freq

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
            num_points = int(np.linalg.norm(x2 - x1) / 0.1)
            for alpha in np.linspace(0, 1, num_points):
                point = (1 - alpha) * x1 + alpha * x2
                point = np.floor(point).astype(int)
                x_idx, y_idx = point[0], point[1]
                if not (0 <= x_idx < self.map_width and 0 <= y_idx < self.map_height):
                    return False
                if self.grid_map[y_idx, x_idx] == 1:
                    return False
            return True

    def grow(self, k, goal, r):
        # Use algorithm from the paper
        for _ in range(k):
            x_random = np.array([
                np.random.uniform(0, self.map_width),
                np.random.uniform(0, self.map_height)
            ])
            x_near = self.find_nearest_neighbour(x_random)
            u = self.select_control(x_near, x_random, step=0.5)
            x_new = self.new_state(x_near, u, step=0.5)

            if self.valid(x_near.states, x_new):
                self.add_state(x_new, x_near)

            if np.linalg.norm(x_new - goal) < r:
                break

        # Ensure kd tree rebuilt at very end
        if self.kd_tree_needs_update:
            self.kd_tree = cKDTree(np.array(self.node_positions))
            self.kd_tree_needs_update = False

    def get_path(self, goal_node):
        path = []
        current = goal_node
        while current is not None:
            path.append(current.states)
            current = current.parent
        return path[::-1]

# ========== RUNNING TEST ==========

grid_map = np.zeros((100, 100), dtype=int)

start = [0, 0]
goal_coords = np.array([90, 90])

# Create RRT
rrt = RRT(x_init=start, grid_map=grid_map, rebuild_freq=100)
rrt.grow(k=5000, goal=goal_coords, r=2.5)

# Find path from start to goal
nearest_to_goal = rrt.find_nearest_neighbour(goal_coords)
path = rrt.get_path(nearest_to_goal)

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
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal_coords, color='blue', label='Goal')
plt.legend()
plt.title("RRT with Grid Map (KDTree Rebuild Frequency)")
plt.axis('equal')
plt.grid(True)
plt.show()

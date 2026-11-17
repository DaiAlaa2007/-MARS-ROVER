"""
Mars Rover Pathfinder - Member 1: AI Navigation Engineer
A* Search & Path Planning

Modules in this file:
  1. MarsMap        -> Grid map (terrain, start, goal, obstacles, dangerous areas)
  2. AStarPlanner   -> A* algorithm + cost calculation (distance, energy, risk)
  3. simulate_mission -> Rover walks the path, discovers new obstacles, re-plans
  4. visualize      -> Matplotlib visualization

Run:  python mars_rover_astar.py
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# 1. Terrain definitions
# ----------------------------------------------------------------------------
FLAT, SAND, ROCK, SLOPE, STORM, OBSTACLE = range(6)

# name, energy multiplier, risk level, color
TERRAIN = {
    FLAT:     ("Flat ground",       1.0, 0.0, "#d9a066"),
    SAND:     ("Soft sand",         2.0, 1.0, "#f0d9a0"),
    ROCK:     ("Rocky terrain",     3.0, 3.0, "#8b6b4a"),
    SLOPE:    ("Steep slope",       4.0, 5.0, "#c0562b"),
    STORM:    ("Sandstorm (danger)", 2.0, 10.0, "#e8c547"),
    OBSTACLE: ("Obstacle (blocked)", 0.0, 0.0, "#2b2b2b"),
}

# Weights for the final cost function
W_ENERGY = 1.0   # importance of energy consumption
W_RISK = 2.0     # importance of terrain risk

# 8-directional movement
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


class MarsMap:
    """Grid map of the Mars surface."""

    def __init__(self, grid, start, goal):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal

    # ---- building ----------------------------------------------------------
    @classmethod
    def generate(cls, rows=30, cols=40, seed=7):
        """Create a random but reproducible Mars surface."""
        rng = np.random.default_rng(seed)
        grid = np.full((rows, cols), FLAT, dtype=int)

        def patch(kind, count, hmax, wmax):
            for _ in range(count):
                h, w = rng.integers(2, hmax + 1), rng.integers(2, wmax + 1)
                r, c = rng.integers(0, rows - h), rng.integers(0, cols - w)
                grid[r:r + h, c:c + w] = kind

        patch(SAND, 6, 6, 9)
        patch(ROCK, 6, 5, 7)
        patch(SLOPE, 4, 4, 6)
        patch(STORM, 2, 6, 8)
        # scattered boulders (impassable)
        for _ in range(int(rows * cols * 0.07)):
            grid[rng.integers(rows), rng.integers(cols)] = OBSTACLE
        # a wall with gaps to make the planning interesting
        wall_c = cols // 2
        grid[3:rows - 3, wall_c] = OBSTACLE
        grid[rows // 3, wall_c] = FLAT
        grid[2 * rows // 3, wall_c] = ROCK

        start, goal = (rows - 3, 2), (2, cols - 3)
        for (r, c) in (start, goal):
            grid[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = FLAT
        return cls(grid, start, goal)

    # ---- queries -----------------------------------------------------------
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked(self, r, c):
        return self.grid[r, c] == OBSTACLE

    def add_obstacle(self, r, c):
        self.grid[r, c] = OBSTACLE

    def cell_factor(self, r, c):
        """Cost per unit distance inside one cell = energy + risk (weighted)."""
        _, energy, risk, _ = TERRAIN[self.grid[r, c]]
        return W_ENERGY * energy + W_RISK * risk

    def copy(self):
        return MarsMap(self.grid.copy(), self.start, self.goal)


# ----------------------------------------------------------------------------
# 2. A* planner
# ----------------------------------------------------------------------------
@dataclass
class PlanResult:
    path: list = field(default_factory=list)
    total_cost: float = math.inf
    distance: float = 0.0
    energy: float = 0.0
    risk: float = 0.0
    expanded: int = 0

    @property
    def found(self):
        return bool(self.path)


class AStarPlanner:
    """
    g(n): accumulated cost  = sum( step_distance * (W_E*energy + W_R*risk) )
    h(n): straight-line (octile) distance to the goal.
          Minimum cell factor is 1 (flat ground), so h never overestimates
          -> admissible and consistent -> A* returns the optimal path.
    """

    @staticmethod
    def heuristic(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)  # octile distance

    @staticmethod
    def step_cost(mars, a, b):
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        return dist * (mars.cell_factor(*a) + mars.cell_factor(*b)) / 2

    def neighbors(self, mars, node):
        r, c = node
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not mars.in_bounds(nr, nc) or mars.is_blocked(nr, nc):
                continue
            # don't cut corners between two obstacles
            if dr and dc and (mars.is_blocked(r + dr, c) or mars.is_blocked(r, c + dc)):
                continue
            yield (nr, nc)

    def plan(self, mars, start=None, goal=None):
        start = start or mars.start
        goal = goal or mars.goal
        open_heap = [(self.heuristic(start, goal), 0.0, start)]
        g = {start: 0.0}
        parent = {start: None}
        closed = set()

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                return self._build_result(mars, parent, cur, g_cur, len(closed))
            for nb in self.neighbors(mars, cur):
                new_g = g_cur + self.step_cost(mars, cur, nb)
                if new_g < g.get(nb, math.inf):
                    g[nb] = new_g
                    parent[nb] = cur
                    heapq.heappush(open_heap, (new_g + self.heuristic(nb, goal), new_g, nb))
        return PlanResult(expanded=len(closed))  # no path

    def _build_result(self, mars, parent, node, cost, expanded):
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        res = PlanResult(path=path, total_cost=cost, expanded=expanded)
        for a, b in zip(path, path[1:]):
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            res.distance += d
            res.energy += d * (TERRAIN[mars.grid[a]][1] + TERRAIN[mars.grid[b]][1]) / 2
            res.risk += d * (TERRAIN[mars.grid[a]][2] + TERRAIN[mars.grid[b]][2]) / 2
        return res


# ----------------------------------------------------------------------------
# 3. Dynamic re-planning
# ----------------------------------------------------------------------------
def simulate_mission(true_map, planner, hidden_obstacles, sensor_range=2):
    """
    The rover only knows `known_map`. `hidden_obstacles` exist in reality but
    are discovered when they come within `sensor_range` cells of the rover.
    When one blocks the current path, the rover re-plans from where it stands.
    Returns: (trajectory, list_of_plans, discovered_obstacles)
    """
    # the rover's map does NOT include the hidden obstacles
    known = true_map.copy()
    hidden = set(hidden_obstacles)

    plans = [planner.plan(known)]
    if not plans[0].found:
        return [], plans, []
    path, pos_idx = plans[0].path, 0
    trajectory = [path[0]]
    discovered = []

    while path[pos_idx] != known.goal:
        pos = path[pos_idx]
        # sensor scan
        new_found = [h for h in list(hidden)
                     if max(abs(h[0] - pos[0]), abs(h[1] - pos[1])) <= sensor_range]
        for h in new_found:
            known.add_obstacle(*h)
            hidden.discard(h)
            discovered.append(h)
        # does a new obstacle block the rest of the path?
        if any(h in path[pos_idx:] for h in new_found):
            new_plan = planner.plan(known, start=pos)
            plans.append(new_plan)
            if not new_plan.found:
                print("No safe path to goal after re-planning!")
                break
            path, pos_idx = new_plan.path, 0
        pos_idx += 1
        trajectory.append(path[pos_idx])
    return trajectory, plans, discovered


# ----------------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------------
def visualize(mars, plans, discovered=(), trajectory=None, title="", save_as=None):
    cmap = ListedColormap([TERRAIN[k][3] for k in range(len(TERRAIN))])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(mars.grid, cmap=cmap, vmin=0, vmax=len(TERRAIN) - 1)

    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    for i, plan in enumerate(plans):
        if not plan.found:
            continue
        ys, xs = zip(*plan.path)
        label = "Initial path" if i == 0 else f"Re-planned path #{i}"
        ax.plot(xs, ys, color=colors[i % 4], lw=2.5 if i == 0 else 2,
                ls="--" if i == 0 and len(plans) > 1 else "-", label=label)
    if trajectory:
        ys, xs = zip(*trajectory)
        ax.plot(xs, ys, color="black", lw=1, alpha=.6, label="Actual rover trajectory")
    if discovered:
        ys, xs = zip(*discovered)
        ax.scatter(xs, ys, marker="X", s=140, c="red", edgecolors="white", label="New obstacle detected")

    ax.scatter(mars.start[1], mars.start[0], marker="o", s=220, c="lime", edgecolors="black", label="Start (landing)", zorder=5)
    ax.scatter(mars.goal[1], mars.goal[0], marker="*", s=400, c="gold", edgecolors="black", label="Goal (rock sample)", zorder=5)

    ax.set_xticks(np.arange(-.5, mars.cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, mars.rows, 1), minor=True)
    ax.grid(which="minor", color="black", alpha=.08)
    ax.set_title(title or "Mars Rover - A* Path Planning")

    handles, _ = ax.get_legend_handles_labels()
    terrain_patches = [Patch(color=TERRAIN[k][3], label=TERRAIN[k][0]) for k in TERRAIN]
    ax.legend(handles=handles + terrain_patches, loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=9)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=130, bbox_inches="tight")
    return fig


def print_report(name, res):
    print(f"--- {name} ---")
    if not res.found:
        print("No path found.\n")
        return
    print(f"Steps: {len(res.path)}   Nodes expanded: {res.expanded}")
    print(f"Distance: {res.distance:.2f}   Energy: {res.energy:.2f}   "
          f"Risk: {res.risk:.2f}   TOTAL COST: {res.total_cost:.2f}\n")


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    mars = MarsMap.generate(rows=30, cols=40, seed=7)
    planner = AStarPlanner()

    # --- Part 1: normal A* ---
    result = planner.plan(mars)
    print_report("A* on the known map", result)
    visualize(mars, [result], title="A* Path (min energy + risk)", save_as="path_basic.png")

    # --- Part 2: new obstacles appear while the rover is moving ---
    # place 2 hidden boulders on the planned path (ahead of the rover)
    hidden = [result.path[len(result.path) // 3], result.path[len(result.path) // 2]]
    traj, plans, found = simulate_mission(mars, planner, hidden)
    for i, p in enumerate(plans):
        print_report("Initial plan" if i == 0 else f"Re-plan #{i}", p)
    final_map = mars.copy()
    for h in found:
        final_map.add_obstacle(*h)
    visualize(final_map, plans, found, traj,
              title="Dynamic re-planning when new obstacles appear",
              save_as="path_replanned.png")

    plt.show()

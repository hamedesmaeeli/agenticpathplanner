import numpy as np
import heapq

def plan(grid, start, goal):
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return []
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def compute_clearance(grid):
        obstacles = np.argwhere(grid == 1)
        clearance_map = np.full_like(grid, float('inf'))

        for obstacle in obstacles:
            x, y = obstacle
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= x + dx < rows and 0 <= y + dy < cols:
                        clearance_map[x + dx, y + dy] = min(clearance_map[x + dx, y + dy], np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2))

        return clearance_map

    clearance_map = compute_clearance(grid)

    start_state = (start[0], start[1])
    goal_state = (goal[0], goal[1])

    if grid[start_state] == 1 or grid[goal_state] == 1:
        return []

    best_cost = {start_state: 0}
    came_from = {}
    counter = itertools.count()

    open_set = [(0, next(counter), start_state)]

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal_state:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_state)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                step_cost = 1 if abs(dr) + abs(dc) == 1 else np.sqrt(2)
                risk = 0 if clearance_map[neighbor] == float('inf') else 1 / (1 + clearance_map[neighbor])
                edge_cost = step_cost * (1 + 0.5 * risk)

                new_cost = best_cost[current] + edge_cost

                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, next(counter), neighbor))

    return []
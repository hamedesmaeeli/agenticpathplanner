import numpy as np
import heapq

def plan(grid, start, goal):
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return []
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def calculate_clearance(obstacles):
        clearance_map = np.full_like(grid, float('inf'))
        for r, c in obstacles:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if 0 <= r + dr < rows and 0 <= c + dc < cols:
                        clearance_map[r + dr, c + dc] = min(clearance_map[r + dr, c + dc], np.sqrt(dr**2 + dc**2))
        return clearance_map

    obstacles = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 1]
    clearance_map = calculate_clearance(obstacles)

    start_state = (start[0], start[1])
    goal_state = (goal[0], goal[1])

    if start_state == goal_state:
        return [start_state]

    best_cost = {start_state: 0}
    came_from = {}
    counter = itertools.count()

    open_set = []
    heapq.heappush(open_set, (0, next(counter), start_state))

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal_state:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_state)
            return path[::-1]

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0], neighbor[1]] == 0:
                step_cost = 1 if (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] else np.sqrt(2)
                risk = 0 if clearance_map[neighbor[0], neighbor[1]] == float('inf') else 1 / (1 + clearance_map[neighbor[0], neighbor[1]])
                edge_cost = step_cost * (1 + 0.5 * risk)
                new_cost = best_cost[current] + edge_cost

                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, next(counter), neighbor))

    return []
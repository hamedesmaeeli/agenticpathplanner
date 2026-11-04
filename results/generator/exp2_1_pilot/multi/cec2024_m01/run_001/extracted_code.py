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
        for obstacle in obstacles:
            r, c = obstacle
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if (dr, dc) != (0, 0):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            clearance_map[nr, nc] = min(clearance_map[nr, nc], np.sqrt(dr**2 + dc**2))
        return clearance_map

    obstacles = [(i, j) for i in range(rows) for j in range(cols) if grid[i, j] == 1]
    clearance_map = calculate_clearance(obstacles)

    start_state = (start[0], start[1])
    goal_state = (goal[0], goal[1])

    if clearance_map[start_state[0], start_state[1]] == float('inf') or clearance_map[goal_state[0], goal_state[1]] == float('inf'):
        return []

    best_cost = {start_state: 0}
    came_from = {}
    counter = 0

    open_set = [(0, counter, start_state)]

    while open_set:
        current_cost, _, current_state = heapq.heappop(open_set)

        if current_state == goal_state:
            path = []
            while current_state in came_from:
                path.append(current_state)
                current_state = came_from[current_state]
            path.append(start_state)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current_state[0] + dr, current_state[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0], neighbor[1]] == 0:
                step_cost = 1 if (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0)] else np.sqrt(2)
                risk = 0 if clearance_map[neighbor[0], neighbor[1]] == float('inf') else 1 / (1 + clearance_map[neighbor[0], neighbor[1]])
                edge_cost = step_cost * (1 + 0.5 * risk)
                new_cost = current_cost + edge_cost

                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(open_set, (new_cost, counter, neighbor))
                    came_from[neighbor] = current_state

    return []
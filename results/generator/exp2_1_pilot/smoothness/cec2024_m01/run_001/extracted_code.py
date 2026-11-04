import numpy as np
import heapq

def plan(grid, start, goal):
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return []
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return []
    if grid[start] == 1 or grid[goal] == 1:
        return []

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    direction_names = ['NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE']

    start_state = (start[0], start[1], 0, 0)
    goal_state = (goal[0], goal[1])

    best_cost = {start_state: (0.0, 0.0)}
    came_from = {}
    open_set = [(0.0, start_state)]
    counter = itertools.count()

    while open_set:
        _, current_state = heapq.heappop(open_set)

        if current_state[:2] == goal_state:
            path = []
            while current_state in came_from:
                path.append(current_state[:2])
                current_state = came_from[current_state]
            path.append(start_state[:2])
            path.reverse()
            return path

        for dr, dc in directions:
            nr, nc = current_state[0] + dr, current_state[1] + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                new_direction = (dr, dc)
                prev_direction = (current_state[2], current_state[3])

                turn_angle = 0.0
                if prev_direction != (0, 0):
                    cos_theta = np.clip(np.dot(prev_direction, new_direction) / 
                                        (np.linalg.norm(prev_direction) * np.linalg.norm(new_direction)), -1, 1)
                    turn_angle = np.arccos(cos_theta)

                new_cost = best_cost[current_state][0] + turn_angle
                if (nr, nc, dr, dc) not in best_cost or new_cost < best_cost[(nr, nc, dr, dc)][0]:
                    best_cost[(nr, nc, dr, dc)] = (new_cost, best_cost[current_state][1] + np.sqrt(dr**2 + dc**2))
                    heapq.heappush(open_set, (best_cost[(nr, nc, dr, dc)][0], (nr, nc, dr, dc), next(counter)))
                    came_from[(nr, nc, dr, dc)] = current_state

    return []
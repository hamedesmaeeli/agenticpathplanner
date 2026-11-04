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
    turn_costs = {('N', 'N'): 0.0, ('N', 'NE'): np.pi/2, ('N', 'E'): np.pi, ('N', 'SE'): 3*np.pi/2,
                   ('NE', 'N'): np.pi/2, ('NE', 'NE'): 0.0, ('NE', 'E'): np.pi/2, ('NE', 'SE'): np.pi,
                   ('E', 'N'): np.pi, ('E', 'NE'): np.pi/2, ('E', 'E'): 0.0, ('E', 'SE'): np.pi/2,
                   ('SE', 'N'): 3*np.pi/2, ('SE', 'NE'): np.pi, ('SE', 'E'): np.pi/2, ('SE', 'SE'): 0.0,
                   ('S', 'N'): -np.pi, ('S', 'NE'): -3*np.pi/2, ('S', 'E'): -np.pi/2, ('S', 'SE'): 0.0,
                   ('SW', 'N'): -np.pi/2, ('SW', 'NE'): 0.0, ('SW', 'E'): np.pi/2, ('SW', 'SE'): np.pi,
                   ('W', 'N'): -np.pi, ('W', 'NE'): -3*np.pi/2, ('W', 'E'): -np.pi/2, ('W', 'SE'): 0.0}

    start_state = (start[0], start[1], 0, 0)
    goal_state = (goal[0], goal[1])

    open_set = []
    heapq.heappush(open_set, (0.0, start_state))
    came_from = {}
    best_cost = {start_state: (0.0, 0)}

    counter = 0
    while open_set:
        current_cost, current_state = heapq.heappop(open_set)
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
                new_state = (nr, nc, dr, dc)
                turn_angle = turn_costs.get((direction_names[(current_state[2]+4)%8], direction_names[(dr+4)%8]), np.pi)
                total_turn_angle, total_length = best_cost[current_state]
                new_total_turn_angle = total_turn_angle + turn_angle
                new_total_length = total_length + 1 if dr == current_state[2] and dc == current_state[3] else total_length + np.sqrt(2)

                if new_state not in best_cost or (new_total_turn_angle, new_total_length) < best_cost[new_state]:
                    best_cost[new_state] = (new_total_turn_angle, new_total_length)
                    came_from[new_state] = current_state
                    heapq.heappush(open_set, (new_total_turn_angle + new_total_length, counter, new_state))
                    counter += 1

    return []
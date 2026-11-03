import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    direction_costs = {d: np.sqrt(2) if d[0] != 0 and d[1] != 0 else 1 for d in directions}

    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def get_neighbors(state):
        row, col, dr, dc = state
        neighbors = []
        for d in directions:
            new_row, new_col = row + d[0], col + d[1]
            if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row, new_col] == 0:
                cost = direction_costs[d]
                if dr != 0 or dc != 0:
                    turn_angle = np.arccos(np.dot((dr, dc), d) / (np.linalg.norm((dr, dc)) * np.linalg.norm(d)))
                    cost += turn_angle
                neighbors.append(((new_row, new_col, d[0], d[1]), cost))
        return neighbors

    start_state = (start + (0, 0), 0)
    goal_state = (goal + (0, 0), 0)

    open_set = []
    heapq.heappush(open_set, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}

    while open_set:
        _, current_state = heapq.heappop(open_set)

        if current_state[0] == goal_state[0]:
            path = []
            while current_state in came_from:
                path.append(current_state[0])
                current_state = came_from[current_state]
            path.append(start_state[0])
            return path[::-1]

        for neighbor, cost in get_neighbors(current_state):
            tentative_g_score = g_score[current_state] + cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_state
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor[0], goal_state[0])
                heapq.heappush(open_set, (f_score, neighbor))

    return []
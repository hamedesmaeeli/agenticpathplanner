import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    direction_costs = {dir: np.sqrt(2) if dir[0] != 0 and dir[1] != 0 else 1 for dir in directions}

    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def get_neighbors(state):
        row, col, dr, dc = state
        neighbors = []
        for d in directions:
            nr, nc = row + d[0], col + d[1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                cost = direction_costs[d] + (np.arctan2(d[0], d[1]) - np.arctan2(dr, dc)) % (2 * np.pi)
                neighbors.append((nr, nc, d[0], d[1], state[4] + cost))
        return neighbors

    start_state = (*start, 0, 0, 0)
    open_set = []
    heapq.heappush(open_set, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}

    while open_set:
        _, current_state = heapq.heappop(open_set)

        if current_state[:2] == goal:
            path = []
            while current_state in came_from:
                path.append((current_state[0], current_state[1]))
                current_state = came_from[current_state]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current_state):
            neighbor_state = neighbor
            tentative_g_score = g_score[current_state] + neighbor_state[4]

            if neighbor_state[:2] not in g_score or tentative_g_score < g_score[neighbor_state]:
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor_state[:2], goal)
                heapq.heappush(open_set, (f_score, neighbor_state))

    return []
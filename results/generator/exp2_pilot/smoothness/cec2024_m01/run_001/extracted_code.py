import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    direction_to_angle = {
        (-1, -1): np.pi / 4, (-1, 0): 0, (-1, 1): -np.pi / 4,
        (0, -1): np.pi / 2, (0, 1): -np.pi / 2,
        (1, -1): -3 * np.pi / 4, (1, 0): -np.pi, (1, 1): -5 * np.pi / 4
    }

    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def cost(current, next, previous_dr, previous_dc):
        dr = next[0] - current[0]
        dc = next[1] - current[1]
        if previous_dr is None or previous_dc is None:
            return heuristic(current, next)
        angle_change = direction_to_angle[(dr, dc)] - direction_to_angle[(previous_dr, previous_dc)]
        return np.sqrt(dr ** 2 + dc ** 2) + abs(angle_change)

    def is_valid_move(state):
        row, col, _, _ = state
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col] == 0

    open_set = []
    heapq.heappush(open_set, (0, start + (None, None)))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_state = heapq.heappop(open_set)
        current, previous_dr, previous_dc = current_state

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current, _, _ = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if not is_valid_move((neighbor, dr, dc)):
                continue
            tentative_g_score = g_score[current] + cost(current, neighbor, previous_dr, previous_dc)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_state
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (g_score[neighbor], neighbor + (dr, dc)))

    return []
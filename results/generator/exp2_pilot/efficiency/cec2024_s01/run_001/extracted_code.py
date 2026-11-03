import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    heuristic = lambda r, c: max(abs(r - goal[0]), abs(c - goal[1])) + (np.sqrt(2) - 1) * min(abs(r - goal[0]), abs(c - goal[1]))

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

    open_set = []
    heapq.heappush(open_set, (heuristic(start[0], start[1]), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if is_valid(neighbor[0], neighbor[1]):
                tentative_g_score = g_score[current] + 1 if abs(dr) == abs(dc) else g_score[current] + np.sqrt(2)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor[0], neighbor[1])
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

    return []
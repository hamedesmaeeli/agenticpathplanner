import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(r, c):
        dx = abs(r - goal[0])
        dy = abs(c - goal[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                tentative_g_score = g_score[current] + (np.sqrt(2) if abs(dr) != 0 and abs(dc) != 0 else 1)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor[0], neighbor[1])
                    heapq.heappush(open_set, (f_score, neighbor))

    return []
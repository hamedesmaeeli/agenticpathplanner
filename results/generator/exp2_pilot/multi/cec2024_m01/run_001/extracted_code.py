import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def clearance(p):
        x, y = p
        if grid[x, y] == 1:
            return float('inf')
        min_dist = float('inf')
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 or dy != 0:
                    ox, oy = x + dx, y + dy
                    if grid[ox, oy] == 1:
                        dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                        min_dist = min(min_dist, dist)
        return min_dist

    def edge_cost(u, v):
        dx, dy = v[0] - u[0], v[1] - u[1]
        d = np.sqrt(dx**2 + dy**2)
        c = clearance(v)
        return d * (1 + 0.5 / (1 + c))

    def reconstruct_path(came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                tentative_g_score = g_score[current] + edge_cost(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))

    return []
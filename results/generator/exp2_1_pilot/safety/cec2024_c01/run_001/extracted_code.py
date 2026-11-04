import heapq
import math
import numpy as np

def plan(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    clearance = np.zeros_like(grid, dtype=float)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                continue
            min_dist = float('inf')
            for orow in range(max(0, r-2), min(rows, r+3)):
                for ocol in range(max(0, c-2), min(cols, c+3)):
                    if grid[orow, ocol] == 1:
                        dist = math.sqrt((r-orow)**2 + (c-ocol)**2)
                        if dist < min_dist:
                            min_dist = dist
            clearance[r, c] = min_dist

    if clearance[start] == float('inf') or clearance[goal] == float('inf'):
        return []

    best = {start: (clearance[start], 0)}
    pq = [(-clearance[start], 0, start)]
    counter = itertools.count()

    while pq:
        neg_bottleneck, length, u = heapq.heappop(pq)
        bottleneck = -neg_bottleneck

        if u == goal:
            path = []
            while u in best:
                path.append(u)
                u = best[u][1]
            return path[::-1]

        for dr, dc in directions:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < rows and 0 <= v[1] < cols and grid[v] == 0:
                new_bottleneck = min(bottleneck, clearance[v])
                new_length = length + math.sqrt(dr**2 + dc**2)
                if v not in best or (new_bottleneck > best[v][0] or (new_bottleneck == best[v][0] and new_length < best[v][1])):
                    best[v] = (new_bottleneck, new_length)
                    heapq.heappush(pq, (-new_bottleneck, new_length, v, next(counter)))

    return []
import heapq
import numpy as np

def plan(grid, start, goal):
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return []
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return []
    
    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def clearance(cell):
        r, c = cell
        min_dist = np.inf
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 1:
                    dist = np.sqrt((i - r) ** 2 + (j - c) ** 2)
                    if dist < min_dist:
                        min_dist = dist
        return min_dist
    
    start_clearance = clearance(start)
    goal_clearance = clearance(goal)
    
    if start_clearance == 0 or goal_clearance == 0:
        return []
    
    best = {(start[0], start[1]): (start_clearance, 0)}
    pq = [(-start_clearance, 0, start)]
    counter = 0
    
    while pq:
        bottleneck, length, u = heapq.heappop(pq)
        r, c = u
        
        if u == goal:
            path = []
            while u in best:
                path.append(u)
                u = best[u][1]
            return path[::-1]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                v = (nr, nc)
                new_bottleneck = min(best[u][0], clearance(v))
                new_length = length + np.sqrt(dr**2 + dc**2)
                if v not in best or (new_bottleneck > best[v][0] or (new_bottleneck == best[v][0] and new_length < best[v][1])):
                    best[v] = (new_bottleneck, new_length)
                    counter += 1
                    heapq.heappush(pq, (-new_bottleneck, new_length, v))
    
    return []
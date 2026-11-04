import numpy as np
import heapq

def plan(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    
    if grid[start] == 1 or grid[goal] == 1:
        return []
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def clearance(cell):
        min_dist = float('inf')
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    dist = euclidean_distance((cell[0], cell[1]), (r, c))
                    if dist < min_dist:
                        min_dist = dist
        return min_dist
    
    start_clearance = clearance(start)
    goal_clearance = clearance(goal)
    
    best = {start: (start_clearance, 0)}
    pq = [(-start_clearance, 0, start)]
    counter = 0
    
    while pq:
        bottleneck, length, u = heapq.heappop(pq)
        bottleneck = -bottleneck
        
        if u == goal:
            path = []
            while u in best:
                path.append(u)
                u = best[u][1]
            return path[::-1]
        
        for dr, dc in directions:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < rows and 0 <= v[1] < cols and grid[v] == 0:
                new_bottleneck = min(bottleneck, clearance(v))
                new_length = length + 1
                if v not in best or (new_bottleneck > best[v][0] or (new_bottleneck == best[v][0] and new_length < best[v][1])):
                    best[v] = (new_bottleneck, new_length)
                    heapq.heappush(pq, (-new_bottleneck, new_length, v))
                    counter += 1
    
    return []
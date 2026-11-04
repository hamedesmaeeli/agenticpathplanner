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
    
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def clearance(cell):
        r, c = cell
        min_dist = float('inf')
        for obstacle in obstacles:
            dist = euclidean_distance((r, c), obstacle)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    obstacles = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 1]
    
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
            while u in predecessors:
                path.append(u)
                u = predecessors[u]
            path.append(start)
            path.reverse()
            return path
        
        for dr, dc in directions:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < grid.shape[0] and 0 <= v[1] < grid.shape[1] and grid[v] == 0:
                v_clearance = clearance(v)
                candidate_bottleneck = min(bottleneck, v_clearance)
                candidate_length = length + (np.sqrt(2) if dr != 0 and dc != 0 else 1)
                
                if v not in best or (candidate_bottleneck > best[v][0] or 
                                      (candidate_bottleneck == best[v][0] and candidate_length < best[v][1])):
                    best[v] = (candidate_bottleneck, candidate_length)
                    predecessors[v] = u
                    heapq.heappush(pq, (-candidate_bottleneck, candidate_length, v))
                    counter += 1
    
    return []
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
        if grid[cell] == 1:
            return float('inf')
        distances = [euclidean_distance(cell, obstacle) for obstacle in obstacles]
        return min(distances)
    
    obstacles = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 1]
    clearance_map = {(r, c): clearance((r, c)) for r in range(rows) for c in range(cols)}
    
    start_state = (start[0], start[1])
    goal_state = (goal[0], goal[1])
    
    if start_state == goal_state:
        return [start_state]
    
    best_cost = {start_state: 0}
    came_from = {}
    counter = 0
    
    open_set = []
    heapq.heappush(open_set, (best_cost[start_state], counter, start_state))
    
    while open_set:
        current_cost, _, current_state = heapq.heappop(open_set)
        
        if current_state == goal_state:
            path = []
            while current_state in came_from:
                path.append(current_state)
                current_state = came_from[current_state]
            path.append(start_state)
            return path[::-1]
        
        for dr, dc in directions:
            neighbor = (current_state[0] + dr, current_state[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                step_cost = 1 if (dr == 0 or dc == 0) else np.sqrt(2)
                risk = 0 if clearance_map[neighbor] == float('inf') else 1 / (1 + clearance_map[neighbor])
                edge_cost = step_cost * (1 + 0.5 * risk)
                new_cost = best_cost[current_state] + edge_cost
                
                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    came_from[neighbor] = current_state
                    heapq.heappush(open_set, (new_cost, counter, neighbor))
                    counter += 1
    
    return []
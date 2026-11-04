import numpy as np
import heapq

def plan(grid, start, goal):
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def is_valid(state):
        row, col, _, _ = state
        return 0 <= row < rows and 0 <= col < cols and grid[row, col] == 0
    
    def get_neighbors(state):
        row, col, prev_dr, prev_dc = state
        neighbors = []
        for dr, dc in directions:
            if (dr, dc) != (-prev_dr, -prev_dc):  # Avoid reversing direction immediately
                nr, nc = row + dr, col + dc
                if is_valid((nr, nc, dr, dc)):
                    neighbors.append((nr, nc, dr, dc))
        return neighbors
    
    def heuristic(state):
        row, col, _, _ = state
        return np.sqrt((row - goal[0])**2 + (col - goal[1])**2)
    
    start_state = (start[0], start[1], 0, 0)
    if not is_valid(start_state) or not is_valid(goal):
        return []
    
    best_cost = {start_state: (0.0, 0)}
    came_from = {}
    counter = 0
    open_set = [(heuristic(start_state), counter, start_state)]
    
    while open_set:
        _, _, current_state = heapq.heappop(open_set)
        
        if current_state[:2] == goal:
            path = []
            while current_state in came_from:
                path.append(current_state[:2])
                current_state = came_from[current_state]
            path.append(start_state[:2])
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current_state):
            nr, nc, dr, dc = neighbor
            prev_dr, prev_dc = current_state[2], current_state[3]
            
            if prev_dr == 0 and prev_dc == 0:
                turn_angle = 0.0
            else:
                dot_product = prev_dr * dr + prev_dc * dc
                norm_prev = np.sqrt(prev_dr**2 + prev_dc**2)
                norm_curr = np.sqrt(dr**2 + dc**2)
                cos_theta = max(-1, min(1, dot_product / (norm_prev * norm_curr)))
                turn_angle = np.arccos(cos_theta)
            
            new_cost = best_cost[current_state][0] + turn_angle
            if neighbor not in best_cost or new_cost < best_cost[neighbor][0]:
                best_cost[neighbor] = (new_cost, best_cost[current_state][1] + heuristic(neighbor))
                came_from[neighbor] = current_state
                counter += 1
                heapq.heappush(open_set, (best_cost[neighbor][0], counter, neighbor))
    
    return []
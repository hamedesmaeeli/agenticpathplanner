import numpy as np
import heapq

def plan(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    
    if grid[start] == 1 or grid[goal] == 1:
        return []
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def compute_clearance(grid):
        obstacles = np.argwhere(grid == 1)
        clearance_map = np.full_like(grid, float('inf'))
        
        for obstacle in obstacles:
            x, y = obstacle
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= x + dx < rows and 0 <= y + dy < cols:
                        clearance_map[x + dx, y + dy] = min(clearance_map[x + dx, y + dy], np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2))
        
        return clearance_map
    
    clearance_map = compute_clearance(grid)
    
    def heuristic(state):
        return clearance_map[state[0], state[1]]
    
    start_state = start
    goal_state = goal
    
    if start_state == goal_state:
        return [start_state]
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start_state), 0, start_state))
    came_from = {}
    cost_so_far = {start_state: 0}
    
    while open_set:
        current_cost, _, current = heapq.heappop(open_set)
        
        if current == goal_state:
            break
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor] == 1:
                continue
            
            new_cost = cost_so_far[current] + 1 + 0.5 / (1 + clearance_map[neighbor])
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(open_set, (priority, len(came_from), neighbor))
                came_from[neighbor] = current
    
    if goal_state not in came_from:
        return []
    
    path = []
    current = goal_state
    while current != start_state:
        path.append(current)
        current = came_from[current]
    path.append(start_state)
    path.reverse()
    
    return path
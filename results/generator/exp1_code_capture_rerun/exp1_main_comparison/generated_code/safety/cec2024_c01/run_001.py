```python
import numpy as np
import heapq

def plan(grid, start, goal):
    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0
    
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
            path.reverse()
            return path
        
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            tentative_g_score = g_score[current] + 1
            
            if is_valid(neighbor[0], neighbor[1]):
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor] + heuristic(neighbor, goal), neighbor))
    
    return []
```
import numpy as np
import heapq

def plan(grid, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return []

    rows, cols = grid.shape
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def clearance(p):
        x, y = p
        min_dist = float('inf')
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 1:
                    dist = np.sqrt((nx - x)**2 + (ny - y)**2)
                    min_dist = min(min_dist, dist)
        return min_dist

    def is_valid(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols and grid[p[0], p[1]] == 0

    start_clearance = clearance(start)
    goal_clearance = clearance(goal)

    if start_clearance == 0 or goal_clearance == 0:
        return []

    open_set = []
    heapq.heappush(open_set, (-start_clearance, 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for direction in directions:
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            if is_valid(next_pos):
                new_cost = cost_so_far[current] + heuristic(current, next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = -min(clearance(p) for p in [current, next_pos]) * 1000000 + new_cost
                    heapq.heappush(open_set, (priority, new_cost, next_pos))
                    came_from[next_pos] = current

    return []
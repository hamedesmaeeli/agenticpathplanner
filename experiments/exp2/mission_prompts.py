"""Mission-specific prompts for Exp2 (standalone; does not alter Exp1)."""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT = r"""You are an expert algorithm designer for grid-based autonomous path planning.
Return ONLY executable Python source code. No Markdown fences or explanations.
Define exactly: def plan(grid, start, goal):
- grid is a 2-D numpy-compatible array: 0=free, 1=obstacle.
- start and goal are (row, col) integer tuples.
- Return a list of (row, col) integer tuples including start and goal; return [] if unreachable.

Movement contract:
- Eight-connected moves: dr,dc each in {-1,0,1}, excluding (0,0).
- Destination must be in bounds and free. Diagonal corner cutting is permitted.
- Orthogonal distance is 1; diagonal distance is sqrt(2).

Reliability contract (mandatory):
1. Use explicit, consistent state tuples. Never unpack a state into a different number of variables.
2. Keep accumulated objective costs in dictionaries keyed by the COMPLETE search state; use full-state dictionary keys.
3. Do not store an accumulated cost inside a state and then add it again.
4. Check bounds BEFORE indexing grid or any grid-shaped array.
5. Use heapq entries with a monotonically increasing counter before non-orderable state data.
6. Reconstruct the path from a predecessor dictionary; verify start and goal are included.
7. Handle start==goal and blocked/out-of-bounds endpoints.
8. Use only Python standard library and numpy. Explicitly import every module used; allowed: math, heapq, itertools, collections.
9. No file access, network, external services, or project imports.
10. Implement the exact mission objective below. Returning any feasible shortest path is not sufficient unless it optimizes that mission.
"""

MISSION_PROMPTS: dict[str, str] = {
    "efficiency": r"""MISSION: EFFICIENCY — EXACT SHORTEST GEOMETRIC PATH

Use A* with the octile heuristic or Dijkstra. Edge cost is hypot(dr,dc):
1 for orthogonal and sqrt(2) for diagonal. The octile heuristic is
max(dx,dy)+(sqrt(2)-1)*min(dx,dy). Manhattan distance is NOT admissible here.
Maintain g_score[(row,col)] and came_from[(row,col)]. On each neighbor:
step = sqrt(2) if dr != 0 and dc != 0 else 1.0
tentative = g_score[current] + step
Do not reverse this condition. Do not add clearance or turn penalties.
Return a minimum geometric-length path.
""",
    "safety": r"""MISSION: SAFETY — MAXIMIZE THE PATH BOTTLENECK CLEARANCE

For each free cell p, clearance[p] is the minimum Euclidean distance from p's
cell centre to any obstacle cell centre (infinity if no obstacles).
Maximize C_min(path)=min(clearance[p] for p on path). Among paths with equal
maximum C_min, minimize geometric path length.

Required method: widest-path Dijkstra with lexicographic labels.
For each cell store best[(r,c)] = (bottleneck, length), initialized at start
as (clearance[start],0). Extending u->v gives candidate=(min(bottleneck,
clearance[v]), length+step). Candidate is better if its bottleneck is larger;
if equal, its length is smaller. Use a max-priority queue (negative bottleneck,
then length, then counter, then cell). Do not use ordinary shortest-path A*.
Compute clearance using obstacle coordinates and Euclidean distances; do not
index neighboring cells without checking bounds. A target clearance of 2.0 is
desirable but not mandatory if infeasible.
""",
    "smoothness": r"""MISSION: SMOOTHNESS — MINIMIZE TOTAL TURNING ANGLE, THEN LENGTH

For consecutive movement vectors a and b, turning angle is
acos(clamp(dot(a,b)/(norm(a)*norm(b)),-1,1)). Straight continuation costs 0;
a reversal costs pi. Minimize the SUM of all interior turning angles. Among
paths with equal total turning angle, minimize geometric path length.

Required search state is exactly (row,col,prev_dr,prev_dc). Use initial state
(start_row,start_col,0,0), where (0,0) means no previous direction and causes
zero initial turn cost. A transition to (nr,nc) creates new state
(nr,nc,dr,dc). Store best_cost[state]=(total_turn_angle,total_length), and
came_from keyed by the COMPLETE state. Compare labels lexicographically.
For noninitial states compute the turn angle from (prev_dr,prev_dc) to (dr,dc).
Use Dijkstra over these nonnegative lexicographic edge costs. Do not put g-cost
inside the state. Do not key g_score only by (row,col), because direction matters.
Reconstruct full state sequence and return only its (row,col) coordinates.
""",
    "multi": r"""MISSION: MULTI-CRITERIA — EXACT LENGTH/CLEARANCE WEIGHTED COST

For every free cell p, clearance[p] is minimum Euclidean distance to any
obstacle-cell centre (infinity if no obstacles). For every move u->v:
step = 1 for orthogonal, sqrt(2) for diagonal
risk = 0 if clearance[v] is infinity else 1/(1+clearance[v])
edge_cost = step * (1 + 0.5*risk)
Minimize the SUM of edge_cost using Dijkstra. Use destination-cell clearance
exactly as written and coefficient 0.5. Do not optimize length alone or
clearance alone. Do not approximate clearance with a local window.

Compute the clearance map safely from obstacle coordinates (vectorized or
nested loops); never read grid[r+dr,c+dc] before checking both bounds. Use
best_cost[(row,col)] and came_from[(row,col)]. Every edge cost is nonnegative,
so Dijkstra is appropriate and requires no heuristic.
""",
}


def build_user_prompt(data: PromptInput, mission: str) -> str:
    if mission not in MISSION_PROMPTS:
        raise ValueError(f"Unknown mission: {mission}")
    grid_block = (
        f"\nGrid (S=start, G=goal, .=free, #=obstacle):\n{data.ascii_grid}\n"
        if data.ascii_grid else ""
    )
    return (
        f"Map: {data.map_id} ({data.height}x{data.width}; {data.difficulty})\n"
        f"Start: {data.start}; Goal: {data.goal}\n"
        f"Obstacle ratio: {data.obstacle_ratio:.2%}\n"
        f"{grid_block}\n{MISSION_PROMPTS[mission]}\n"
        "Before returning code, check state tuple arity, full-state dictionary keys, "
        "cost accumulation, and all array bounds. Output only Python source."
    )


@dataclass(frozen=True)
class PromptInput:
    map_id: str
    height: int
    width: int
    difficulty: str
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacle_ratio: float
    ascii_grid: str | None = None




def build_user_prompt(data: PromptInput, mission: str) -> str:
    if mission not in MISSION_PROMPTS:
        raise ValueError(f"Unknown mission: {mission}")
    grid_block = (
        f"\nGrid (S=start, G=goal, .=free, #=obstacle):\n{data.ascii_grid}\n"
        if data.ascii_grid else ""
    )
    return (
        f"Map: {data.map_id} ({data.height}x{data.width}; {data.difficulty})\n"
        f"Start: {data.start}; Goal: {data.goal}\n"
        f"Obstacle ratio: {data.obstacle_ratio:.2%}\n"
        f"{grid_block}\n{MISSION_PROMPTS[mission]}\n"
        "Before returning code, check state tuple arity, full-state dictionary keys, "
        "cost accumulation, and all array bounds. Output only Python source."
    )

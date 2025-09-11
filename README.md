# Blind Search in Large-Scale Problem Spaces

This project implements blind search algorithms (BFS, DFS, Depth-Limited Search, Iterative Deepening)
and compares them against informed search methods (A* and Greedy Search).  
Three problem domains are tested:
- Maze navigation
- 8-Puzzle
- N-Queens

## Algorithms Implemented
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Depth-Limited Search (DLS)
- Iterative Deepening Search (IDS)
- Greedy Best-First Search
- A* Search

## Requirements
- Python 3.10+
- matplotlib
- numpy
- tracemalloc (built-in)
 Install dependencies:

## **Experiments**
```bash
pip install -r requirements.txt
python experiments/run_experiments.py


---
import tracemalloc
import time
from collections import deque
from typing import Any, Callable, List, Optional, Set, Tuple
import matplotlib.pyplot as plt

class Node:
    """
    Represents a node in a search tree.
    """
    def __init__(
        self,
        state: Any,
        parent: Optional["Node"]=None,
        action: Optional[Any]=None,
        depth: int=0,
        cost: float=0.0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.cost = cost

    def path(self) -> List[Any]:
        """
        Returns the path from the root to this node.
        """
        node, p = self, []
        while node:
            p.append(node.state)
            node = node.parent
        return list(reversed(p))

def measure(func: Callable, *args, **kwargs) -> dict:
    """
    Measure execution time, peak memory (MB), nodes expanded, and result.

    Args:
        func: Function to measure.
        *args, **kwargs: Arguments to pass to func.

    Returns:
        dict with keys: 'time', 'mem', 'nodes', 'result'
    """
    tracemalloc.start()
    start = time.time()
    result, nodes = func(*args, **kwargs)
    end = time.time()
    mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Peak memory in MB
    tracemalloc.stop()
    return {
        "time": end - start,
        "mem": mem,
        "nodes": nodes,
        "result": result
    }

def bfs(
    start: Any,
    goal: Any,
    neighbors: Callable[[Any], List[Any]]
) -> Tuple[Optional[List[Any]], int]:
    """
    Breadth-First Search.

    Args:
        start: Initial state.
        goal: Goal state.
        neighbors: Function returning neighbors of a state.

    Returns:
        (path, nodes_expanded)
    """
    frontier = deque([Node(start)])
    explored: Set[Any] = set()
    in_frontier: Set[Any] = {start}
    nodes = 0

    while frontier:
        node = frontier.popleft()
        nodes += 1
        in_frontier.discard(node.state)
        if node.state == goal:
            return node.path(), nodes
        explored.add(node.state)
        for n in neighbors(node.state):
            if n not in explored and n not in in_frontier:
                frontier.append(Node(n, node))
                in_frontier.add(n)
    return None, nodes

def dfs(
    start: Any,
    goal: Any,
    neighbors: Callable[[Any], List[Any]],
    limit: int=1000
) -> Tuple[Optional[List[Any]], int]:
    """
    Depth-First Search with optional depth limit.

    Args:
        start: Initial state.
        goal: Goal state.
        neighbors: Function returning neighbors of a state.
        limit: Maximum search depth.

    Returns:
        (path, nodes_expanded)
    """
    frontier = [Node(start)]
    explored: Set[Any] = set()
    in_frontier: Set[Any] = {start}
    nodes = 0

    while frontier:
        node = frontier.pop()
        nodes += 1
        in_frontier.discard(node.state)
        if node.state == goal:
            return node.path(), nodes
        if node.depth < limit:
            explored.add(node.state)
            for n in neighbors(node.state):
                if n not in explored and n not in in_frontier:
                    frontier.append(Node(n, node, depth=node.depth + 1))
                    in_frontier.add(n)
    return None, nodes

def solve_n_queens(n: int) -> Tuple[List[List[int]], int]:
    """
    Solves the N-Queens problem using backtracking.
    
    Args:
        n: Number of queens and size of the board.
        
    Returns:
        Tuple of:
            - List of solutions (each solution is a list of column positions for each row)
            - Number of nodes (recursive calls made)
    """
    solutions = []
    nodes = 0

    def is_valid(state: List[int], row: int, col: int) -> bool:
        for r in range(row):
            c = state[r]
            if c == col or abs(c - col) == row - r:
                return False
        return True

    def backtrack(row: int, state: List[int]):
        nonlocal nodes
        nodes += 1
        if row == n:
            solutions.append(state[:])
            return
        for col in range(n):
            if is_valid(state, row, col):
                state.append(col)
                backtrack(row + 1, state)
                state.pop()

    backtrack(0, [])
    return solutions, nodes

def puzzle_search(
    start: Tuple[int, ...],
    goal: Tuple[int, ...],
    neighbors: Callable[[Tuple[int, ...]], List[Tuple[int, ...]]]
) -> Tuple[Optional[List[Tuple[int, ...]]], int]:
    """
    General puzzle search using BFS (suitable for sliding puzzles like 8-puzzle).

    Args:
        start: Initial state (e.g., tuple representing puzzle).
        goal: Goal state.
        neighbors: Function returning neighbor states.

    Returns:
        (path to solution, nodes expanded)
    """
    frontier = deque([Node(start)])
    explored: Set[Tuple[int, ...]] = set()
    in_frontier: Set[Tuple[int, ...]] = {start}
    nodes = 0

    while frontier:
        node = frontier.popleft()
        nodes += 1
        in_frontier.discard(node.state)
        if node.state == goal:
            return node.path(), nodes
        explored.add(node.state)
        for n in neighbors(node.state):
            if n not in explored and n not in in_frontier:
                frontier.append(Node(n, node))
                in_frontier.add(n)
    return None, nodes

def plot_complexity(results, title):
    """
    Plot time and memory complexity for multiple runs.
    """
    x = [r['input_size'] for r in results]
    times = [r['time'] for r in results]
    mems = [r['mem'] for r in results]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Time (s)', color=color)
    ax1.plot(x, times, color=color, marker='o', label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Memory (MB)', color=color)
    ax2.plot(x, mems, color=color, marker='x', label='Memory')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Maze Search Example
    def neighbors_example(state):
        return [state - 1, state + 1]

    start_state, goal_state = 0, 5
    bfs_result = measure(bfs, start_state, goal_state, neighbors_example)
    dfs_result = measure(dfs, start_state, goal_state, neighbors_example)

    print("=== BFS Result ===")
    print(f"Path: {bfs_result['result']}")
    print(f"Nodes Expanded: {bfs_result['nodes']}")
    print(f"Time: {bfs_result['time']:.6f}s")
    print(f"Peak Memory: {bfs_result['mem']:.6f} MB\n")

    print("=== DFS Result ===")
    print(f"Path: {dfs_result['result']}")
    print(f"Nodes Expanded: {dfs_result['nodes']}")
    print(f"Time: {dfs_result['time']:.6f}s")
    print(f"Peak Memory: {dfs_result['mem']:.6f} MB\n")

    # N-Queens Results (varying n for graphs)
    nqueens_results = []
    for n in range(4, 10):
        nqueens_result = measure(solve_n_queens, n)
        nqueens_results.append({
            'input_size': n,
            'time': nqueens_result['time'],
            'mem': nqueens_result['mem'],
            'nodes': nqueens_result['nodes'],
            'num_solutions': len(nqueens_result['result'])
        })
    print("=== N-Queens Results ===")
    for r in nqueens_results:
        print(f"n={r['input_size']}, Solutions={r['num_solutions']}, Nodes={r['nodes']}, Time={r['time']:.6f}s, Peak Memory={r['mem']:.6f}MB")
    plot_complexity(nqueens_results, "N-Queens Time & Space Complexity")

    # Puzzle Search Example (8-puzzle, varying start states for graphs)
    def puzzle_neighbors(state):
        idx = state.index(0)
        moves = []
        row, col = divmod(idx, 3)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_idx = new_row * 3 + new_col
                new_state = list(state)
                new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[idx]
                moves.append(tuple(new_state))
        return moves

    puzzle_results = []
    # Generate start states with increasing distance from goal for demo (here just one for brevity)
    goal_puzzle = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    start_puzzles = [
        (1, 2, 3, 4, 5, 6, 7, 8, 0),
        (1, 2, 3, 4, 5, 6, 0, 7, 8),
        (1, 2, 3, 4, 5, 6, 8, 7, 0)
    ]
    for idx, start_puzzle in enumerate(start_puzzles):
        puzzle_result = measure(puzzle_search, start_puzzle, goal_puzzle, puzzle_neighbors)
        puzzle_results.append({
            'input_size': idx + 1,
            'time': puzzle_result['time'],
            'mem': puzzle_result['mem'],
            'nodes': puzzle_result['nodes'],
            'solution_found': puzzle_result['result'] is not None
        })
        print(f"Puzzle {idx+1}: Nodes={puzzle_result['nodes']}, Time={puzzle_result['time']:.6f}s, Peak Memory={puzzle_result['mem']:.6f}MB")
    plot_complexity(puzzle_results, "Puzzle Search Time & Space Complexity")

    # Maze Search complexity plot (varying goal distance)
    maze_results = []
    for goal_state in range(2, 12, 2):
        bfs_result = measure(bfs, start_state, goal_state, neighbors_example)
        maze_results.append({
            'input_size': goal_state,
            'time': bfs_result['time'],
            'mem': bfs_result['mem'],
            'nodes': bfs_result['nodes']
        })
        print(f"Maze Goal={goal_state}: Nodes={bfs_result['nodes']}, Time={bfs_result['time']:.6f}s, Peak Memory={bfs_result['mem']:.6f}MB")
    plot_complexity(maze_results, "Maze Search (BFS) Time & Space Complexity")




```


## Result
1.) Maze Problem

<img width="252" height="240" alt="image" src="https://github.com/user-attachments/assets/2fa9381f-8a3f-4a08-8411-93dd471d65a2" />


2.) N-Queen Problem

<img width="702" height="160" alt="image" src="https://github.com/user-attachments/assets/e339676a-b0c5-478d-a003-1fe4a05e950b" />

3.) Puzzle Search

<img width="637" height="84" alt="image" src="https://github.com/user-attachments/assets/e8ee9988-5059-4a89-8616-b144c10ea3ec" />


## Graphical Comparison of Time & Space complexity

1.) Maze Problem

<img width="818" height="696" alt="image" src="https://github.com/user-attachments/assets/3b5e981d-f1fe-4e55-aaf7-1e1d291678a6" />

2.) N-Queen Problem

<img width="803" height="574" alt="image" src="https://github.com/user-attachments/assets/44b61a09-7474-425f-ba8e-503c447533f7" />

3.) Puzzle Search

<img width="841" height="568" alt="image" src="https://github.com/user-attachments/assets/d68939ae-4e41-4337-b7af-1ff6cf96862a" />





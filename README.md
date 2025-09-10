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

## 3. `experiments/utils.py`
```python
import tracemalloc, time

class Node:
    def __init__(self, state, parent=None, action=None, depth=0, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.cost = cost

    def path(self):
        node, p = self, []
        while node:
            p.append(node.state)
            node = node.parent
        return list(reversed(p))

def measure(func, *args, **kwargs):
    """Measure execution time, memory, and nodes expanded."""
    tracemalloc.start()
    start = time.time()
    result, nodes = func(*args, **kwargs)
    end = time.time()
    mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    return {
        "time": end - start,
        "mem": mem,
        "nodes": nodes,
        "result": result
    }




# Maze Search
from collections import deque
from experiments.utils import Node

def bfs(start, goal, neighbors):
    frontier = deque([Node(start)])
    explored = set()
    nodes = 0
    while frontier:
        node = frontier.popleft()
        nodes += 1
        if node.state == goal:
            return node.path(), nodes
        explored.add(node.state)
        for n in neighbors(node.state):
            if n not in explored:
                frontier.append(Node(n, node))
    return None, nodes

def dfs(start, goal, neighbors, limit=1000):
    frontier = [Node(start)]
    explored = set()
    nodes = 0
    while frontier:
        node = frontier.pop()
        nodes += 1
        if node.state == goal:
            return node.path(), nodes
        if node.depth < limit:
            explored.add(node.state)
            for n in neighbors(node.state):
                if n not in explored:
                    frontier.append(Node(n, node, depth=node.depth+1))
    return None, nodes
```
##


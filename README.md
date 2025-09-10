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

## **Experiments**

# Maze Search
- from collections import deque
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

    - **Result**


# Puzzle Search

-from experiments.utils import Node

def manhattan(state, goal):
    d = 0
    for i, val in enumerate(state):
        if val == 0: continue
        xi, yi = divmod(i, 3)
        xg, yg = divmod(goal.index(val), 3)
        d += abs(xi - xg) + abs(yi - yg)
    return d

def astar(start, goal, neighbors, h):
    frontier = [Node(start, cost=0)]
    explored = {}
    nodes = 0
    while frontier:
        frontier.sort(key=lambda n: n.cost + h(n.state, goal))
        node = frontier.pop(0)
        nodes += 1
        if node.state == goal:
            return node.path(), nodes
        explored[node.state] = node.cost
        for s in neighbors(node.state):
            cost = node.cost + 1
            if s not in explored or cost < explored[s]:
                frontier.append(Node(s, node, depth=node.depth+1, cost=cost))
    return None, nodes
- Result


# N-Queen Puzzle

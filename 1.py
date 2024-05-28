# a)

import heapq

def best_first_search(graph, start, goal, heuristic):
    # Priority queue for exploring nodes based on heuristic
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start], start))
    visited = set()
    parent = {start: None}

    while priority_queue:
        current_heuristic, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            break

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (heuristic[neighbor], neighbor))
                parent[neighbor] = current_node

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path

# Example graph
# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F', 'G'],
#     'D': [],
#     'E': [],
#     'F': [],
#     'G': []
# }

# # Example heuristic values (assumed for demonstration)
# heuristic = {
#     'A': 6,
#     'B': 4,
#     'C': 4,
#     'D': 0,
#     'E': 2,
#     'F': 3,
#     'G': 1
# }

# start = 'A'
# goal = 'D'

# path = best_first_search(graph, start, goal, heuristic)
# print("Best First Search Path:", path)
#if u want user input
def create_graph():
    graph = {}
    while True:
        node = input("Enter a node (or 'done' to finish): ").strip()
        if node == 'done':
            break
        neighbors = input(f"Enter neighbors of node {node} separated by space: ").strip().split()
        graph[node] = neighbors
    return graph

def create_heuristic():
    heuristic = {}
    while True:
        node = input("Enter a node (or 'done' to finish): ").strip()
        if node == 'done':
            break
        heuristic[node] = int(input(f"Enter heuristic value for node {node}: ").strip())
    return heuristic

# Input graph and heuristic from user
print("Enter the graph:")
graph = create_graph()

print("\nEnter the heuristic values:")
heuristic = create_heuristic()

start = input("\nEnter the start node: ").strip()
goal = input("Enter the goal node: ").strip()

# Perform best first search
path = best_first_search(graph, start, goal, heuristic)
print("\nBest First Search Path:", path)



# b)

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Doors']
z = dataset['Price']

ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap="jet")
ax.set_title("3D Surface Plot")

plt.show()

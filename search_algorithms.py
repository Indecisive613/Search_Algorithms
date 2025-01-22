from collections import deque 
import heapq 

def print_forest(forest): 
    for row in forest: 
        print(" ".join(row)) 
    print() 

class SearchAgent: 
    def __init__(self, start, goal, grid): 
        self.start = start 
        self.goal = goal 
        self.grid = grid 
        self.rows = len(grid) 
        self.cols = len(grid[0]) 

    def is_valid(self, position): 
        row, col = position 
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != '#' 

    def get_cost(self, position): 
        cell = self.grid[position[0]][position[1]] 
        if cell == 'f': 
            return 3 
        elif cell == 'F': 
            return 5 
        return 1 

    def print_path(self, path): 
        temp_grid = [row[:] for row in self.grid]  # Create a copy of the grid 
        for row, col in path[1:-1]:  # Avoid overriding start and goal 
            temp_grid[row][col] = 'P' 
        for row in temp_grid: 
            print(" ".join(row))

    def get_valid_neighbors(self, pos):
        # Return a list of neighbors that can be visited.
        row, col = pos
        valid_neighbors = []
        # Check top neighbor
        if self.is_valid((row-1, col)):
            valid_neighbors.append((row-1,col))
        # Check bottom neighbor
        if self.is_valid((row+1, col)):
            valid_neighbors.append((row+1,col))
        # Check left neighbor
        if self.is_valid((row, col-1)):
            valid_neighbors.append((row,col-1))
        # Check right neighbor
        if self.is_valid((row, col+1)):
            valid_neighbors.append((row,col+1))
        return valid_neighbors

    def find_start(self):
        # Return the position of the start marker
        # Start searching from the top left corner
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == 'S':
                    return (row,col)
        return None

    def find_end(self):
        # Return the position of the goal
        # Start searching from the bottom right corner
        for row in range(self.rows-1, -1, -1):
            for col in range(self.cols-1, -1, -1):
                if self.grid[row][col] == '*':
                    return (row,col)
        return None

    def print_forest(self, path):
        # Prints the contents of the grid, showing the path taken with Ps
        for row in range(self.rows):
            for col in range(self.cols):
                symbolToPrint = self.grid[row][col]
                if (row,col) in path:
                    symbolToPrint = 'P'
                print(symbolToPrint, end=' ')
            print()

    def bfs(self): 
        # TODO Implement BFS logic: return exploration steps, path cost, and path length, or None if no path is found. 
        return [-1,-1,-1]

    def dfs(self): 
        # Implements DFS logic to find the goal: returns exploration steps, path cost, and path length, or None if no path is found.
        startRow, startCol = self.find_start()
        goalRow, goalCol = self.find_end()
        # Start with the starting node in the fringe
        # Format is (<row>, <column>, <full path to this position>, <cost of the full path>)
        fringe = [(startRow, startCol, [(startRow,startCol)], 0)]
        visited = []
        exploration_steps = 0
        while len(fringe) > 0:
            # Explore the last node in the fringe (DFS uses a LIFO)
            row, col, path, cost = fringe.pop()
            visited.append((row, col))
            exploration_steps += 1
            # Check if the current node is the goal
            if row == goalRow and col == goalCol:
                self.print_forest(path)
                return len(path)-1, exploration_steps, cost
            # Attempt to add all neighbors to the fringe
            for r,c in self.get_valid_neighbors((row,col)):
                if (r,c) not in visited:
                    # Add the unvisited neighbor to the fringe
                    tempPath = path.copy()
                    tempPath.append((r,c))
                    fringe.append((r, c, tempPath, cost + self.get_cost((r,c)))) # Reminder that fringe has items of (<row>, <column>, <path>, <cost>)
        return None
 
    def ucs(self): 
        # Implement UCS logic: return exploration steps, path cost, and path length, or None if no path is found. 
        start_Row, start_Col = self.find_start()
        goal_Row, goal_Col = self.find_end()
        fringe = [(0, start_Row, start_Col, [(start_Row, start_Col)])]
        visited = {}
        exploration_steps = 0
        while len(fringe) > 0:
            # Explore the node with the minimum cost (UCS uses a min-heap)
            cost, row, col, path = heapq.heappop(fringe)
            if (row, col) in visited and visited[(row, col)] <= cost:
                continue
            # Otherwise, mark the node as visited with its cost
            visited[(row, col)] = cost
            exploration_steps += 1
            # Check if the current node is the goal
            if row == goal_Row and col == goal_Col:
                self.print_forest(path)
                return len(path) - 1, exploration_steps, cost
            # Add valid neighbors
            for r, c in self.get_valid_neighbors((row, col)):
                new_cost = cost + self.get_cost((r, c))
                temp_Path = path.copy()  # Create a new path forneighbor
                temp_Path.append((r, c))
                heapq.heappush(fringe, (new_cost, r, c, temp_Path))
        return None


    def astar(self, heuristic=None): 
        # Implement A* logic: return exploration steps, path cost, and path length, or None if no path is found. 
        return [-1,-1,-1]
        if heuristic is None: 
            def heuristic(pos): 
                return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]) 

def test_search_agent(agent): 
    results = {} 
    print("\n--- BFS ---") 
    path_length, exploration_steps, cost_length = agent.bfs() 
    results['BFS'] = (path_length, exploration_steps, cost_length) 
    print(f"Path Length: {path_length} steps") 
    print(f"Exploration Steps: {exploration_steps}") 
    print(f"Cost Length: {cost_length}") 
    print("\n--- DFS ---") 
    path_length, exploration_steps, cost_length = agent.dfs() 
    results['DFS'] = (path_length, exploration_steps, cost_length) 
    print(f"Path Length: {path_length} steps") 
    print(f"Exploration Steps: {exploration_steps}") 
    print(f"Cost Length: {cost_length}") 
    print("\n--- UCS ---") 
    path_length, exploration_steps, cost_length = agent.ucs() 
    results['UCS'] = (path_length, exploration_steps, cost_length) 
    print(f"Path Length: {path_length} steps") 
    print(f"Exploration Steps: {exploration_steps}") 
    print(f"Cost Length: {cost_length}") 
    print("\n--- A* ---") 
    path_length, exploration_steps, cost_length = agent.astar(lambda pos: abs(pos[0] - agent.goal[0]) + 
abs(pos[1] - agent.goal[1])) 
    results['A*'] = (path_length, exploration_steps, cost_length) 
    print(f"Path Length: {path_length} steps") 
    print(f"Exploration Steps: {exploration_steps}") 
    print(f"Cost Length: {cost_length}") 
    return results 

Agents = {} 
forest0 = [['S', '.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'], 
    ['#', '#', '.', '#', '#', '.', '.', '#', '#', '#', '#', '.', '#', '.', '.'], 
    ['.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '#', '.', '#', '.', '.'], 
    ['.', '#', '#', '#', '.', '.', '#', '.', '#', '#', '#', '.', '#', '#', '#'], 
    ['.', '#', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '#'], 
    ['.', '#', '.', '#', '#', '#', '.', '#', '#', '.', '#', '#', '#', '.', '#'], 
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'], 
    ['#', '#', '#', '.', '#', '#', '#', '#', '#', '#', '.', '#', '#', '.', '#'], 
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], 
    ['#', '.', '#', '#', '#', '#', '#', '#', '.', '#', '#', '.', '#', '#', '#'], 
    ['#', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.'], 
    ['#', '.', '#', '#', '.', '#', '.', '#', '#', '#', '#', '.', '#', '#', '#'], 
    ['#', '.', '#', '.', '.', '#', '.', '.', '.', '.', '#', '.', '.', '.', '#'], 
    ['#', '.', '.', '.', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'], 
    ['#', '#', '#', '#', '#', '#', '#', '.', '.', '.', '#', '#', '#', '#', '*']] 
start0 = (0, 0) 
goal0 = (14, 14) 
Agents[0] = SearchAgent(start0, goal0, forest0) 

forest1 = [["S", ".", ".", ".", "#", "#", "#", ".", ".", ".", ".", ".", ".", ".", "."], 
    ["#", "#", ".", "#", "#", ".", "f", "#", "#", "#", "#", ".", "F", ".", "."], 
    [".", ".", ".", "#", ".", ".", ".", ".", ".", "f", "#", ".", ".", "F", "."], 
    [".", "#", "#", "#", ".", ".", "#", ".", "#", "#", "#", ".", "#", "#", "#"], 
    [".", "#", ".", ".", ".", "#", ".", ".", ".", ".", ".", ".", ".", ".", "#"], 
    [".", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#", "#", "#", ".", "#"], 
    [".", "f", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "F", ".", "#"], 
    ["#", "#", "#", ".", "#", "#", "#", "#", "#", "#", ".", "#", "#", ".", "#"], 
    [".", ".", ".", ".", ".", ".", ".", ".", "f", ".", ".", ".", ".", ".", "."], 
    ["#", ".", "#", "#", "#", "#", "#", "#", ".", "#", "#", ".", "F", "#", "#"], 
    ["#", ".", ".", ".", ".", "#", ".", ".", ".", ".", ".", ".", "f", ".", "."], 
    ["#", ".", "#", "#", ".", "#", ".", "#", "#", "#", "#", ".", "#", "#", "#"], 
    ["#", ".", "#", ".", ".", "#", ".", ".", ".", ".", "#", ".", "F", ".", "#"], 
    ["#", ".", ".", ".", "#", ".", ".", "#", ".", ".", ".", ".", ".", ".", "F"], 
    ["#", "#", "#", "#", "#", "#", "#", ".", "F", ".", "#", "#", "#", "f", "*"]] 
start1 = (0, 0) 
goal1 = (14, 14) 
Agents[1] = SearchAgent(start1, goal1, forest1) 

forest2 = [["S", ".", ".", ".", ".", "#", ".", ".", ".", "f", ".", "#", ".", ".", "."], 
    [".", "F", ".", "#", ".", "#", ".", "#", ".", ".", ".", ".", ".", "F", "."], 
    [".", "#", ".", ".", "f", "#", ".", ".", "#", ".", ".", "F", ".", ".", "."], 
    ["f", ".", ".", "#", ".", "#", "#", ".", ".", "#", "#", "#", "#", "#", "#"], 
    [".", ".", ".", "#", ".", "f", ".", ".", ".", ".", ".", ".", "f", ".", "."], 
    [".", ".", "#", ".", "#", ".", "#", "#", "#", "#", ".", ".", "F", ".", "#"], 
    [".", ".", ".", ".", "#", ".", ".", ".", ".", "f", ".", ".", ".", ".", "."], 
    [".", ".", "F", ".", "#", ".", "#", "#", "#", ".", "#", "#", "#", ".", "#"], 
    ["#", ".", "f", ".", ".", ".", ".", ".", ".", "#", "f", ".", ".", ".", "."], 
    ["#", ".", "#", ".", "#", "#", "#", "#", ".", "#", "#", "#", ".", "F", "#"], 
    ["f", ".", "#", ".", ".", ".", ".", ".", "#", ".", ".", ".", ".", ".", "."], 
    ["#", ".", "#", ".", "#", ".", "#", ".", "#", ".", "#", "#", "#", "#", "#"], 
    [".", ".", ".", ".", "#", "F", "#", ".", ".", ".", "f", ".", "f", ".", "."], 
    [".", "#", ".", ".", ".", "f", ".", "#", "#", ".", ".", ".", ".", "F", "F"], 
    [".", "#", ".", ".", "#", "#", ".", "#", ".", ".", ".", ".", ".", "f", "*"]] 
start2 = (0, 0) 
goal2 = (14, 14) 
Agents[2] = SearchAgent(start2, goal2, forest2) 

forest3 = [["S", ".", ".", "#", ".", ".", ".", "f", ".", "#", ".", ".", ".", ".", "."], 
    ["#", ".", "F", ".", ".", "F", "#", ".", ".", ".", ".", ".", "F", "#", "#"], 
    [".", ".", ".", ".", ".", "#", ".", ".", ".", ".", "#", ".", ".", ".", "."], 
    ["#", "#", "#", ".", "#", ".", ".", ".", "#", "f", ".", ".", ".", ".", "#"], 
    [".", ".", ".", ".", ".", ".", ".", "#", "F", ".", ".", ".", "#", ".", "."], 
    ["#", ".", "#", "#", "#", "#", "#", "#", ".", ".", ".", "#", "#", ".", "."], 
    ["#", ".", ".", ".", ".", ".", ".", "#", "f", ".", ".", ".", "#", ".", "."], 
    [".", ".", "f", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "#"], 
    [".", ".", ".", ".", "#", ".", ".", ".", "#", "#", "#", ".", ".", ".", "."], 
    [".", ".", "F", ".", "#", ".", "F", ".", "#", ".", "f", ".", ".", "#", "#"], 
    ["f", ".", ".", ".", "#", "#", "#", ".", ".", ".", "#", "#", "#", "#", "#"], 
    [".", ".", ".", ".", ".", ".", ".", "#", ".", "F", ".", ".", ".", ".", "#"], 
    [".", ".", "#", ".", "#", "#", ".", "#", "f", ".", ".", ".", ".", "f", "#"], 
    ["#", ".", "#", ".", ".", "F", ".", ".", ".", ".", ".", "#", ".", ".", "F"], 
    ["#", "#", "#", "#", ".", ".", ".", "f", ".", "#", ".", ".", ".", "f", "*"]] 
start3 = (0, 0) 
goal3 = (14, 14) 
Agents[3] = SearchAgent(start3, goal3, forest3) 

for AGENT in Agents: 
    print(f"Forest {AGENT} Solution:") 
    print(test_search_agent(Agents[AGENT])) 

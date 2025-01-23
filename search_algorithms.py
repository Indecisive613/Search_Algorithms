from collections import deque 
import heapq
import pygame
import time 

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

    def bfs(self): 
        startRow, startCol = self.start
        goalRow, goalCol = self.goal
        # Start with the starting node in the fringe
        # Format is (<row>, <column>, <full path to this position>, <cost of the full path>)
        fringe = [(startRow, startCol, [(startRow,startCol)], 0)]
        visited = []
        exploration_steps = 0
        
        while len(fringe) > 0:
            # BFS uses a FIFO
            row, col, path, cost = fringe.pop(0)
            visited.append((row, col))
            exploration_steps += 1
        
        # Check if the current node is the goal
            if row == goalRow and col == goalCol:
                self.print_path(path)
                print("Path: " + str(path))
                return len(path)-1, exploration_steps, cost
        
        # Add valid neighbors to the queue
            for r, c in self.get_valid_neighbors((row, col)):
                if (r, c) not in visited:
                    temp_Path = path.copy()
                    temp_Path.append((r, c))
                    fringe.append((r, c, temp_Path, cost + self.get_cost((r, c))))
        return None

    def dfs(self): 
        # Implements DFS logic to find the goal: returns exploration steps, path cost, and path length, or None if no path is found.
        startRow, startCol = self.start
        goalRow, goalCol = self.goal
        # Start with the starting node in the fringe
        # Format is (<row>, <column>, <full path to this position>, <cost of the full path>)
        fringe = [(startRow, startCol, [(startRow,startCol)], 0)]
        visited = []
        exploration_steps = 0
        while len(fringe) > 0:
            # Explore the last node in the fringe (DFS uses a LIFO)
            row, col, path, cost = fringe.pop()
            # Skip nodes that have already been visited
            if((row,col) in visited):
                continue
            visited.append((row, col))
            exploration_steps += 1
            # Check if the current node is the goal
            if row == goalRow and col == goalCol:
                self.print_path(path)
                print("Path: " + str(path))
                return len(path)-1, exploration_steps, cost
            # Add all neighbors to the fringe
            for r,c in self.get_valid_neighbors((row,col)):
                tempPath = path.copy()
                tempPath.append((r,c))
                fringe.append((r, c, tempPath, cost + self.get_cost((r,c)))) # Reminder that fringe has items of (<row>, <column>, <path>, <cost>)
        return None
 
    def ucs(self): 
        # Implement UCS logic: return exploration steps, path cost, and path length, or None if no path is found. 
        start_Row, start_Col = self.start
        goal_Row, goal_Col = self.goal
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
                self.print_path(path)
                print("Path: " + str(path))
                return len(path) - 1, exploration_steps, cost
            # Add valid neighbors
            for r, c in self.get_valid_neighbors((row, col)):
                new_cost = cost + self.get_cost((r, c))
                temp_Path = path.copy()  # Create a new path forneighbor
                temp_Path.append((r, c))
                heapq.heappush(fringe, (new_cost, r, c, temp_Path))
        return None


    def astar(self, heuristic=None): 
        # Implements A* logic: return exploration steps, path cost, and path length, or None if no path is found.
        # Uses manhatten distance as the heuristic if no heuristic is specified
        if heuristic is None:
            def heuristic(pos):
                return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        start_Row, start_Col = self.start
        goal_Row, goal_Col = self.goal
        # Format: <cost+heuristic> <cost> <row> <col> <path>
        fringe = [(0, 0, start_Row, start_Col, [(start_Row, start_Col)])]
        visited = []
        exploration_steps = 0
        while len(fringe) > 0:
            # Explore the node with the minimum cost_and_heuristic (A* uses a min-heap)
            cost_and_heuristic, cost, row, col, path = heapq.heappop(fringe)
            # Skip nodes that have already been visited
            if((row, col) in visited):
                continue
            visited.append((row,col))
            exploration_steps += 1
            # Check if the current node is the goal
            if row == goal_Row and col == goal_Col:
                self.print_path(path)
                print("Path: " + str(path))
                return len(path) - 1, exploration_steps, cost
            # Add valid neighbors
            for r, c in self.get_valid_neighbors((row, col)):
                new_cost_only = cost + self.get_cost((r, c))
                new_cost_and_heuristic = new_cost_only + heuristic((r, c))
                temp_Path = path.copy()  # Create a new path for neighbor
                temp_Path.append((r, c))
                heapq.heappush(fringe, (new_cost_and_heuristic, new_cost_only, r, c, temp_Path))
        return None

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

def visual_representation(forest, rows, columns, path):
    # Define colours and cell size
    WHITE = (255, 255, 255)  # Locations with no cost
    BLACK = (0, 0, 0)        # Obstacles
    YELLOW = (255, 255, 0)   # Small Fires
    RED = (255, 0, 0)        # Large Fires
    BLUE = (0, 0, 255)       # Marks Pathfinder's Path
    GREEN = (0, 255, 0)      # Marks the goal
    cell_size = 50           # Each cell is 50x50 pixels
    visited = []             # Cells that have already been visisted

    # Set up the display
    pygame.init()
    screen = pygame.display.set_mode((rows * cell_size, columns * cell_size))
    pygame.display.set_caption("Burning Forest")
    
    def draw_grid():
        # Draws the grid and fill it with elements
        for i in range(rows):
            for j in range(columns):
                # Calculate the pixel coordinates of the current cell
                x1, y1 = j * cell_size, i * cell_size

                # Determine the color of the cell
                if (x1,y1) in visited:
                    color = BLUE
                elif forest[i][j] == '.':
                    color = WHITE
                elif forest[i][j] == '#':
                    color = BLACK
                elif forest[i][j] == 'f':
                    color = YELLOW
                elif forest[i][j] == 'F':
                    color = RED
                elif forest[i][j] == '*':
                    color = GREEN
                else:
                    print("ERROR - Undefined object!")
                    color = RED

                # Draw the cell rectangle
                pygame.draw.rect(screen, color, (x1, y1, cell_size, cell_size), 0)
                # Draw the cell outline for visibility
                pygame.draw.rect(screen, BLACK, (x1, y1, cell_size, cell_size), 1)

    def animate_path():
        # Function to animate the path through the grid
        for step in path:
            # Add current position to the visited nodes
            i, j = step
            x1, y1 = j * cell_size, i * cell_size  # Convert grid coordinates to pixel coordinates
            visited.append((x1, y1))

            # Update the display
            draw_grid()
            pygame.display.flip()
            time.sleep(0.5)

    animate_path()
    time.sleep(2)
    pygame.quit()

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

# Call visual_representation(forest, rows, columns, path) to see the visual representation
# BFS
# visual_representation(forest3, 15, 15, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (11, 2), (11, 3), (12, 3), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14)])
# DFS
# visual_representation(forest3, 15, 15, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5), (0, 5), (0, 6), (0, 7), (0, 8), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 12), (2, 13), (3, 13), (3, 12), (3, 11), (3, 10), (3, 9), (4, 9), (4, 10), (5, 10), (5, 9), (5, 8), (6, 8), (6, 9), (6, 10), (6, 11), (7, 11), (7, 12), (7, 13), (8, 13), (8, 12), (8, 11), (9, 11), (9, 10), (9, 9), (10, 9), (10, 8), (10, 7), (9, 7), (9, 6), (9, 5), (8, 5), (8, 6), (8, 7), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (7, 0), (8, 0), (8, 1), (8, 2), (8, 3), (9, 3), (9, 2), (9, 1), (9, 0), (10, 0), (10, 1), (10, 2), (10, 3), (11, 3), (11, 4), (11, 5), (11, 6), (12, 6), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14)])
# UCS
# visual_representation(forest3, 15, 15, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (11, 4), (11, 5), (11, 6), (12, 6), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14)])
# A Star
# visual_representation(forest3, 15, 15, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (11, 4), (11, 5), (11, 6), (12, 6), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14)])
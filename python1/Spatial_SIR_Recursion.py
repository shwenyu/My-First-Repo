import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Increase recursion limit (use with caution)
sys.setrecursionlimit(3000)

class RecursiveVirusSimulation:
    def __init__(self, size=50, infection_rate=0.3, recovery_rate=0.1, vaccination_rate=0.1):
        self.size = size
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.vaccination_rate = vaccination_rate
        
        self.grid = np.zeros((size, size))
        mask = np.random.random((size, size)) < vaccination_rate
        self.grid[mask] = 3
        self.grid[size//2, size//2] = 1
        
        self.fig, self.ax = plt.subplots()
        self.im = plt.imshow(self.grid, cmap='viridis')
        plt.colorbar(self.im)
    
    def update_cell(self, i, j, grid):
        if grid[i,j] == 0:  # Susceptible
            neighbors = []
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if 0 <= i+di < self.size and 0 <= j+dj < self.size:
                        neighbors.append(grid[i+di,j+dj])
            
            infected_neighbors = sum(1 for n in neighbors if n == 1)
            if infected_neighbors > 0 and np.random.random() < self.infection_rate:
                return 1
        elif grid[i,j] == 1 and np.random.random() < self.recovery_rate:
            return 2
        return grid[i,j]
    
    def recursive_update(self, current_step=0, max_steps=100):
        if current_step >= max_steps:
            return
        
        new_grid = self.grid.copy()
        for i in range(self.size):
            for j in range(self.size):
                new_grid[i,j] = self.update_cell(i, j, self.grid)
        
        self.grid = new_grid
        self.im.set_array(self.grid)
        plt.pause(0.1)  # Allow plot to update
        
        # Recursive call for next step
        self.recursive_update(current_step + 1, max_steps)
    
    def simulate(self, num_steps=100):
        plt.ion()  # Interactive mode on
        self.recursive_update(0, num_steps)
        plt.ioff()
        plt.show()

sim = RecursiveVirusSimulation()
sim.simulate(100)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

class VirusSimulation:
    def __init__(self, size=50, infection_rate=0.3, recovery_rate=0.1, vaccination_rate=0.1):
        self.size = size
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.vaccination_rate = vaccination_rate
        
        # States: 0-Susceptible, 1-Infected, 2-Recovered, 3-Vaccinated
        self.grid = np.zeros((size, size))
        
        # Initialize vaccinated population
        mask = np.random.random((size, size)) < vaccination_rate
        self.grid[mask] = 3
        
        # Set initial infection at center
        self.grid[size//2, size//2] = 1
        
        self.fig, self.ax = plt.subplots()
        
    def update(self):
        new_grid = self.grid.copy()
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] == 0:  # Susceptible
                    # Check neighbors
                    neighbors = []
                    for di in [-1,0,1]:
                        for dj in [-1,0,1]:
                            if 0 <= i+di < self.size and 0 <= j+dj < self.size:
                                neighbors.append(self.grid[i+di,j+dj])
                    
                    # Calculate infection probability
                    infected_neighbors = sum(1 for n in neighbors if n == 1)
                    if infected_neighbors > 0 and np.random.random() < self.infection_rate:
                        new_grid[i,j] = 1
                
                elif self.grid[i,j] == 1:  # Infected
                    if np.random.random() < self.recovery_rate:
                        new_grid[i,j] = 2
        
        self.grid = new_grid
        
    def animate(self, num_steps=100):
        self.im = plt.imshow(self.grid, cmap='viridis')
        plt.colorbar(self.im)
        
        def update_fig(frame):
            self.update()
            self.im.set_array(self.grid)
            return [self.im]
        
        anim = animation.FuncAnimation(self.fig, update_fig, frames=num_steps,
                                     interval=100, blit=True)
        plt.show()

sim = VirusSimulation()
sim.animate(100)
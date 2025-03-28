import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')  # Try this backend

class VirusSimulation:
    def __init__(self, size=100, initial_infected_percent=0.01):
        # Grid setup (0=susceptible, 1=infected, 2=recovered, 3=vaccinated)
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Parameters
        self.infection_rate = 0.4
        self.recovery_rate = 0.1
        self.vaccination_rate = 0.002
        self.immunity_period = 90
        
        # Initialize with random infected cells
        num_infected = int(size * size * initial_infected_percent)
        infected_indices = np.random.choice(size*size, num_infected, replace=False)
        infected_coords = np.unravel_index(infected_indices, (size, size))
        self.grid[infected_coords] = 1
        
        # Time tracking
        self.recovery_time = np.zeros((size, size))
        self.vaccination_time = np.zeros((size, size))
        
        # History tracking
        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.vaccinated_history = []
        
        # Setup figure and axes
        self.fig = plt.figure(figsize=(12, 6))
        self.grid_ax = self.fig.add_subplot(121)
        self.plot_ax = self.fig.add_subplot(122)
        
        # Create colormap
        self.cmap = ListedColormap(['lightblue', 'red', 'green', 'yellow'])
        
    def update(self):
        # Create a copy to update
        new_grid = self.grid.copy()
        
        # Update timers for recovered and vaccinated
        self.recovery_time[self.grid == 2] += 1
        self.vaccination_time[self.grid == 3] += 1
        
        # Check for immunity expiration
        expired_recovery = (self.recovery_time >= self.immunity_period) & (self.grid == 2)
        expired_vaccination = (self.vaccination_time >= self.immunity_period) & (self.grid == 3)
        new_grid[expired_recovery | expired_vaccination] = 0
        
        # Process infections and recoveries
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:  # Infected
                    # Check for recovery
                    if np.random.random() < self.recovery_rate:
                        new_grid[i, j] = 2  # Recover
                        self.recovery_time[i, j] = 0
                
                elif self.grid[i, j] == 0:  # Susceptible
                    # Check neighbors for infection
                    i_min, i_max = max(0, i-1), min(self.size, i+2)
                    j_min, j_max = max(0, j-1), min(self.size, j+2)
                    neighbors = self.grid[i_min:i_max, j_min:j_max]
                    
                    infected_neighbors = np.sum(neighbors == 1)
                    if infected_neighbors > 0:
                        infection_prob = self.infection_rate * (infected_neighbors / 8)
                        if np.random.random() < infection_prob:
                            new_grid[i, j] = 1  # Become infected
        
        # Apply vaccination
        susceptible = np.where(new_grid == 0)
        if len(susceptible[0]) > 0:
            num_to_vaccinate = int(self.size * self.size * self.vaccination_rate)
            num_to_vaccinate = min(num_to_vaccinate, len(susceptible[0]))
            
            if num_to_vaccinate > 0:
                indices = np.random.choice(len(susceptible[0]), num_to_vaccinate, replace=False)
                for idx in indices:
                    i, j = susceptible[0][idx], susceptible[1][idx]
                    new_grid[i, j] = 3  # Vaccinate
                    self.vaccination_time[i, j] = 0
        
        self.grid = new_grid
        
        # Update history
        self.susceptible_history.append(np.sum(self.grid == 0))
        self.infected_history.append(np.sum(self.grid == 1))
        self.recovered_history.append(np.sum(self.grid == 2))
        self.vaccinated_history.append(np.sum(self.grid == 3))
    
    def animate_func(self, frame):
        # Update simulation
        self.update()
        
        # Clear axes
        self.grid_ax.clear()
        self.plot_ax.clear()
        
        # Plot grid
        img = self.grid_ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=3)
        self.grid_ax.set_title(f'Day {frame}')
        
        # Add colorbar if it's the first frame
        if frame == 0:
            cbar = self.fig.colorbar(img, ax=self.grid_ax, ticks=[0.4, 1.2, 2.0, 2.8])
            cbar.ax.set_yticklabels(['Susceptible', 'Infected', 'Recovered', 'Vaccinated'])
        
        # Plot history
        x = range(len(self.susceptible_history))
        self.plot_ax.plot(x, self.susceptible_history, 'b-', label='Susceptible')
        self.plot_ax.plot(x, self.infected_history, 'r-', label='Infected')
        self.plot_ax.plot(x, self.recovered_history, 'g-', label='Recovered')
        self.plot_ax.plot(x, self.vaccinated_history, 'y-', label='Vaccinated')
        
        self.plot_ax.set_title('Population Over Time')
        self.plot_ax.set_xlabel('Days')
        self.plot_ax.set_ylabel('Population')
        self.plot_ax.legend(loc='upper right')
        self.plot_ax.grid(True)
        
        # Adjust layout
        self.fig.tight_layout()
        
        return [img]
    
    def run_simulation(self, frames=200):
        ani = animation.FuncAnimation(
            self.fig, 
            self.animate_func, 
            frames=frames,
            interval=100,
            blit=False
        )
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = VirusSimulation(size=100)
    sim.run_simulation(frames=200)

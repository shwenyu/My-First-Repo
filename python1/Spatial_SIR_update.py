import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

class EnhancedVirusSimulation:
    def __init__(self, 
                 size=100,  # Increased size for more realistic simulation
                 infection_rate=0.4,
                 recovery_rate=0.1,
                 initial_vaccination_rate=0.05,  # Lowered initial vaccination
                 daily_vaccination_percentage=0.001,  # 0.1% of population per day
                 immunity_period=90,  # Extended immunity period
                 vaccine_immunity_period=180,  # Extended vaccine immunity
                 vaccine_protection=0.6,  # Reduced protection for more breakthrough cases
                 initial_infected_percentage=0.01):  # 1% initial infected
        
        # Grid size and basic parameters
        self.size = size
        self.total_population = size * size
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.daily_vaccination = int(self.total_population * daily_vaccination_percentage)
        self.immunity_period = immunity_period
        self.vaccine_immunity_period = vaccine_immunity_period
        self.vaccine_protection = vaccine_protection
        
        # Initialize grids
        # States: 0-Susceptible, 1-Infected, 2-Recovered, 3-Vaccinated
        self.grid = np.zeros((size, size))
        self.recovery_time = np.zeros((size, size))
        self.vaccination_time = np.zeros((size, size))
        self.infection_duration = np.zeros((size, size))
        
        # Initialize vaccinated population
        mask = np.random.random((size, size)) < initial_vaccination_rate
        self.grid[mask] = 3
        self.vaccination_time[mask] = 0
        
        # Initialize infected population (randomly distributed)
        initial_infected = int(self.total_population * initial_infected_percentage)
        susceptible_coords = np.where(self.grid == 0)
        infected_indices = np.random.choice(len(susceptible_coords[0]), 
                                         initial_infected, 
                                         replace=False)
        for idx in infected_indices:
            i, j = susceptible_coords[0][idx], susceptible_coords[1][idx]
            self.grid[i,j] = 1
        
        # Setup visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.history = {'susceptible': [], 'infected': [], 
                       'recovered': [], 'vaccinated': []}
        
        # Custom colormap
        colors = ['lightblue', 'red', 'green', 'yellow']
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)
        
    def get_neighbor_states(self, i, j):
        """Get states of neighboring cells using vectorized operations"""
        i_min, i_max = max(0, i-1), min(self.size, i+2)
        j_min, j_max = max(0, j-1), min(self.size, j+2)
        return self.grid[i_min:i_max, j_min:j_max]
    
    def calculate_infection_probability(self, i, j, neighbor_states):
        """Calculate infection probability based on multiple factors"""
        infected_count = np.sum(neighbor_states == 1)
        if infected_count == 0:
            return 0
        
        # Base probability depends on number of infected neighbors and their proximity
        base_prob = self.infection_rate * (infected_count / neighbor_states.size)
        
        # Modify probability based on current state
        if self.grid[i,j] == 3:  # Vaccinated
            # Vaccine protection decreases over time
            time_factor = self.vaccination_time[i,j] / self.vaccine_immunity_period
            current_protection = self.vaccine_protection * (1 - time_factor)
            base_prob *= (1 - current_protection)
        
        return min(base_prob, 0.95)
    
    def update(self):
        """Update the simulation state using vectorized operations where possible"""
        new_grid = self.grid.copy()
        
        # Update time trackers
        self.recovery_time[self.grid == 2] += 1
        self.vaccination_time[self.grid == 3] += 1
        self.infection_duration[self.grid == 1] += 1
        
        # Check immunity expiration
        recovered_expired = (self.recovery_time >= self.immunity_period) & (self.grid == 2)
        vaccine_expired = (self.vaccination_time >= self.vaccine_immunity_period) & (self.grid == 3)
        new_grid[recovered_expired | vaccine_expired] = 0
        
        # Process infections and recoveries
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] in [0, 3]:  # Susceptible or Vaccinated
                    neighbor_states = self.get_neighbor_states(i, j)
                    if np.random.random() < self.calculate_infection_probability(i, j, neighbor_states):
                        new_grid[i,j] = 1
                        self.infection_duration[i,j] = 0
                
                elif self.grid[i,j] == 1:  # Infected
                    # Recovery probability increases with infection duration
                    recovery_prob = self.recovery_rate * (1 + self.infection_duration[i,j]/20)
                    if np.random.random() < min(recovery_prob, 0.95):
                        new_grid[i,j] = 2
                        self.recovery_time[i,j] = 0
                        self.infection_duration[i,j] = 0
        
        # Perform vaccination
        susceptible_coords = np.where(self.grid == 0)
        if len(susceptible_coords[0]) > 0:
            num_to_vaccinate = min(self.daily_vaccination, len(susceptible_coords[0]))
            indices = np.random.choice(len(susceptible_coords[0]), num_to_vaccinate, replace=False)
            for idx in indices:
                i, j = susceptible_coords[0][idx], susceptible_coords[1][idx]
                new_grid[i,j] = 3
                self.vaccination_time[i,j] = 0
        
        self.grid = new_grid
        
    def update_history(self):
        """Update historical data for plotting"""
        counts = {
            'susceptible': np.sum(self.grid == 0),
            'infected': np.sum(self.grid == 1),
            'recovered': np.sum(self.grid == 2),
            'vaccinated': np.sum(self.grid == 3)
        }
        for key, value in counts.items():
            self.history[key].append(value)
        return counts
    
    def animate(self, num_steps=300):
        """Run the animation with dual plots"""
        self.im = self.ax1.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=3)
        plt.colorbar(self.im, ax=self.ax1, 
                    label='State (Blue:Susceptible, Red:Infected, Green:Recovered, Yellow:Vaccinated)')
        
        def update_fig(frame):
            self.update()
            counts = self.update_history()
            
            # Update grid plot
            self.im.set_array(self.grid)
            self.ax1.set_title(f'Day {frame}\nPopulation Distribution')
            
            # Update line plot
            self.ax2.clear()
            for key in self.history.keys():
                self.ax2.plot(self.history[key], label=key.capitalize())
            self.ax2.set_title('Population Trends')
            self.ax2.set_xlabel('Days')
            self.ax2.set_ylabel('Population')
            self.ax2.legend()
            self.ax2.grid(True)
            
            # Add current counts to title
            plt.suptitle(f'Day {frame} - Infected: {counts["infected"]}, '
                        f'Susceptible: {counts["susceptible"]}, '
                        f'Recovered: {counts["recovered"]}, '
                        f'Vaccinated: {counts["vaccinated"]}')
            
            return [self.im]
        
        anim = animation.FuncAnimation(self.fig, update_fig, frames=num_steps,
                                     interval=100, blit=False)
        plt.tight_layout()
        plt.show()

# Run simulation
if __name__ == "__main__":
    sim = EnhancedVirusSimulation()
    sim.animate(300)

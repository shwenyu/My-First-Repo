import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.ndimage import convolve

class EnhancedVirusSimulation:
    def __init__(self, 
                 size=100,
                 base_infection_rate=0.4,
                 recovery_rate=0.1,
                 initial_vaccination_rate=0.01,
                 base_daily_vaccination_percentage=0.002,
                 immunity_period=90,
                 vaccine_immunity_period=180,
                 vaccine_protection=0.85,
                 initial_infected_percentage=0.005,
                 panic_factor_max=3.0,
                 panic_decay=0.995,
                 min_daily_vaccination_rate=0.001,  # Minimum vaccination rate
                 booster_threshold=0.4):  # Threshold for booster eligibility
        
        self.size = size
        self.total_population = size * size
        self.base_infection_rate = base_infection_rate
        self.current_infection_rate = base_infection_rate
        self.recovery_rate = recovery_rate
        self.immunity_period = immunity_period
        self.vaccine_immunity_period = vaccine_immunity_period
        self.vaccine_protection = vaccine_protection
        self.panic_factor = 1.0
        self.panic_factor_max = panic_factor_max
        self.panic_decay = panic_decay
        self.min_daily_vaccination_rate = min_daily_vaccination_rate
        self.booster_threshold = booster_threshold
        
        # Main state grid and timing trackers
        self.grid = np.zeros((size, size))
        self.recovery_time = np.zeros((size, size))
        self.vaccination_time = np.zeros((size, size))
        self.booster_count = np.zeros((size, size))
        self.infection_duration = np.zeros((size, size))
        
        # Reinfection tracking
        self.daily_recovered_reinfections = 0
        self.daily_vaccinated_reinfections = 0
        self.reinfection_count = np.zeros((size, size))
        self.recovered_reinfection_count = np.zeros((size, size))
        self.vaccinated_reinfection_count = np.zeros((size, size))
        
        # Initialize populations
        mask = np.random.random((size, size)) < initial_vaccination_rate
        self.grid[mask] = 3
        
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
        self.history = {
            'susceptible': [], 'infected': [], 'recovered': [], 'vaccinated': [],
            'recovered_reinfected_daily': [], 'vaccinated_reinfected_daily': [],
            'panic_factor': [], 'boosters_given': [], 'total_vaccinated': []
        }
        
        # Custom colormap
        self.colors = ['lightblue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan']
        self.cmap = LinearSegmentedColormap.from_list('custom', self.colors)

    def calculate_immunity_level(self, vaccination_time, booster_count):
        """Calculate current immunity level based on time since vaccination and boosters"""
        base_immunity = max(0, 1 - (vaccination_time / self.vaccine_immunity_period))
        booster_effect = min(0.3, booster_count * 0.1)  # Each booster adds up to 10% protection
        return min(0.95, base_immunity + booster_effect)

    def perform_dynamic_vaccination(self):
        """Enhanced vaccination strategy with continuous coverage and boosters"""
        infection_percentage = np.sum(self.grid == 1) / self.total_population
        
        # Calculate dynamic vaccination rate
        panic_boost = np.sqrt(self.panic_factor)
        infection_boost = 1 + (infection_percentage * 15)
        base_rate = max(self.min_daily_vaccination_rate, 
                       0.002 * panic_boost * infection_boost)
        
        # Cap maximum daily rate
        max_daily_rate = 0.02
        target_vaccination_rate = min(base_rate, max_daily_rate)
        daily_target = int(self.total_population * target_vaccination_rate)
        
        # Identify eligible population for vaccination or boosters
        susceptible_mask = self.grid == 0
        vaccinated_mask = self.grid == 3
        
        # Calculate immunity levels for vaccinated population
        immunity_levels = np.zeros_like(self.grid)
        immunity_levels[vaccinated_mask] = self.calculate_immunity_level(
            self.vaccination_time[vaccinated_mask],
            self.booster_count[vaccinated_mask]
        )
        
        # Identify population needing boosters
        needs_booster = (immunity_levels < self.booster_threshold) & vaccinated_mask
        
        # Calculate vaccination priorities
        hotspots = self.calculate_hotspots()
        vaccination_priority = np.zeros_like(self.grid)
        
        # Priority for new vaccinations
        vaccination_priority[susceptible_mask] = hotspots[susceptible_mask] + 0.2
        
        # Priority for boosters
        vaccination_priority[needs_booster] = (
            (1 - immunity_levels[needs_booster]) * 0.8 +
            hotspots[needs_booster] * 0.2
        )
        
        # Add random component
        random_factor = np.random.random(self.grid.shape) * 0.1
        vaccination_priority += random_factor * (susceptible_mask | needs_booster)
        
        if np.sum(vaccination_priority) > 0:
            vaccination_priority = vaccination_priority / np.sum(vaccination_priority)
            
            # Select vaccination locations
            eligible_count = np.sum(susceptible_mask | needs_booster)
            if eligible_count > 0:
                flat_indices = np.random.choice(
                    self.size * self.size,
                    size=min(daily_target, eligible_count),
                    p=vaccination_priority.flatten(),
                    replace=False
                )
                coords = np.unravel_index(flat_indices, (self.size, self.size))
                
                # Apply vaccinations and boosters
                for i, j in zip(*coords):
                    if self.grid[i, j] == 0:  # New vaccination
                        self.grid[i, j] = 3
                        self.vaccination_time[i, j] = 0
                        self.booster_count[i, j] = 0
                    elif needs_booster[i, j]:  # Booster shot
                        self.vaccination_time[i, j] = 0
                        self.booster_count[i, j] += 1

    def update(self):
        """Update simulation state"""
        new_grid = self.grid.copy()
        self.update_panic_factor()
        
        # Reset daily counters
        self.daily_recovered_reinfections = 0
        self.daily_vaccinated_reinfections = 0
        daily_boosters = 0
        
        # Update time trackers
        self.recovery_time[self.grid == 2] += 1
        self.vaccination_time[self.grid == 3] += 1
        self.infection_duration[self.grid == 1] += 1
        
        # Process infections and recoveries
        for i in range(self.size):
            for j in range(self.size):
                current_state = self.grid[i,j]
                
                if current_state in [0, 2, 3]:  # Susceptible, Recovered, or Vaccinated
                    neighbors = self.get_neighbor_states(i, j)
                    infected_neighbors = np.sum(neighbors == 1)
                    
                    base_prob = self.current_infection_rate * (infected_neighbors / 8)
                    if current_state == 3:  # Vaccinated
                        immunity_level = self.calculate_immunity_level(
                            self.vaccination_time[i,j],
                            self.booster_count[i,j]
                        )
                        base_prob *= (1 - immunity_level)
                    
                    if np.random.random() < base_prob:
                        new_grid[i,j] = 1
                        if current_state == 2:
                            self.daily_recovered_reinfections += 1
                            self.recovered_reinfection_count[i,j] += 1
                        elif current_state == 3:
                            self.daily_vaccinated_reinfections += 1
                            self.vaccinated_reinfection_count[i,j] += 1
                
                elif current_state == 1:  # Infected
                    if np.random.random() < self.recovery_rate:
                        new_grid[i,j] = 2
                        self.recovery_time[i,j] = 0
        
        # Check immunity expiration
        expired_mask = ((self.recovery_time >= self.immunity_period) & (self.grid == 2))
        new_grid[expired_mask] = 0
        
        self.grid = new_grid
        self.perform_dynamic_vaccination()
        
        # Update history
        self.history['boosters_given'].append(daily_boosters)

    def update_panic_factor(self):
        """Update panic factor based on infection rate"""
        infection_percentage = np.sum(self.grid == 1) / self.total_population
        target_panic = 1.0 + (self.panic_factor_max - 1.0) * min(infection_percentage * 10, 1.0)
        self.panic_factor = min(target_panic, 
                              self.panic_factor * (1/self.panic_decay if infection_percentage > 0.05 
                                                 else self.panic_decay))
        self.current_infection_rate = self.base_infection_rate * self.panic_factor

    def get_neighbor_states(self, i, j):
        """Get states of neighboring cells"""
        i_min, i_max = max(0, i-1), min(self.size, i+2)
        j_min, j_max = max(0, j-1), min(self.size, j+2)
        return self.grid[i_min:i_max, j_min:j_max]

    def calculate_hotspots(self):
        """Calculate infection density with wider radius"""
        kernel = np.ones((5, 5))
        kernel[2, 2] = 2
        return convolve((self.grid == 1).astype(float), kernel, mode='constant')

    def update_history(self):
        """Update historical data"""
        counts = {
            'susceptible': np.sum(self.grid == 0),
            'infected': np.sum(self.grid == 1),
            'recovered': np.sum(self.grid == 2),
            'vaccinated': np.sum(self.grid == 3),
            'recovered_reinfected_daily': self.daily_recovered_reinfections,
            'vaccinated_reinfected_daily': self.daily_vaccinated_reinfections,
            'panic_factor': self.panic_factor,
            'total_vaccinated': np.sum((self.grid == 3) | (self.vaccinated_reinfection_count > 0))
        }
        
        for key, value in counts.items():
            self.history[key].append(value)
        return counts

    def animate(self, num_steps=300):
        """Run animation with enhanced visualization"""
        self.im = self.ax1.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=6)
        
        legend_elements = [
            Patch(facecolor=self.colors[0], label='Susceptible'),
            Patch(facecolor=self.colors[1], label='Infected'),
            Patch(facecolor=self.colors[2], label='Recovered'),
            Patch(facecolor=self.colors[3], label='Vaccinated'),
            Patch(facecolor=self.colors[4], label='Reinfected (Recovered)'),
            Patch(facecolor=self.colors[5], label='Reinfected (Vaccinated)'),
            Patch(facecolor=self.colors[6], label='Boosted')
        ]
        self.ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        def update_fig(frame):
            self.update()
            counts = self.update_history()
            
            # Update grid visualization
            display_grid = self.grid.copy()
            display_grid[self.recovered_reinfection_count > 0] = 4
            display_grid[self.vaccinated_reinfection_count > 0] = 5
            display_grid[(self.grid == 3) & (self.booster_count > 0)] = 6
            self.im.set_array(display_grid)
            
            self.ax2.clear()
            
            # Plot main populations
            self.ax2.plot(self.history['susceptible'], 'lightblue', label='Susceptible')
            self.ax2.plot(self.history['infected'], 'red', label='Infected')
            self.ax2.plot(self.history['recovered'], 'green', label='Recovered')
            self.ax2.plot(self.history['vaccinated'], 'yellow', label='Vaccinated')
            
            # Plot reinfections on secondary y-axis
            ax3 = self.ax2.twinx()
            ax3.plot(self.history['recovered_reinfected_daily'], 'orange', 
                    label='Daily Recovered Reinfections', linestyle='--')
            ax3.plot(self.history['vaccinated_reinfected_daily'], 'purple',
                    label='Daily Vaccinated Reinfections', linestyle=':')
            
            # Titles and labels
            self.ax1.set_title(f'Day {frame}\nPanic Factor: {self.panic_factor:.2f}')
            self.ax2.set_title('Population Trends')
            self.ax2.set_xlabel('Days')
            self.ax2.set_ylabel('Population')
            ax3.set_ylabel('Daily Reinfections')
            
            # Combine legends
            lines1, labels1 = self.ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            return [self.im]
        
        anim = animation.FuncAnimation(self.fig, update_fig, frames=num_steps,
                                     interval=100, blit=False)
        plt.tight_layout()
        plt.show()

# Run simulation
if __name__ == "__main__":
    sim = EnhancedVirusSimulation()
    sim.animate(300)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

class VirusSimulation:
    def __init__(self, size=50, 
                 infection_rate=0.3, 
                 recovery_rate=0.1, 
                 initial_vaccination_rate=0.1,
                 daily_vaccination = 100,
                 immunity_period=30,
                 vaccine_immunity_period=60,
                 vaccine_protection=0.7):  # Vaccine reduces infection chance by 70%
        
        # Grid size and basic parameters
        self.size = size
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.daily_vaccination = daily_vaccination
        self.immunity_period = immunity_period
        self.vaccine_immunity_period = vaccine_immunity_period
        self.vaccine_protection = vaccine_protection
        
        # Initialize grids
        # States: 0-Susceptible, 1-Infected, 2-Recovered, 3-Vaccinated
        self.grid = np.zeros((size, size))
        
        # Tracking days since recovery/vaccination for immunity period
        self.recovery_time = np.zeros((size, size))
        self.vaccination_time = np.zeros((size, size))
        self.infection_duration = np.zeros((size, size))
        
        # Initialize vaccinated population
        mask = np.random.random((size, size)) < initial_vaccination_rate
        self.grid[mask] = 3
        self.vaccination_time[mask] = 0
        
        # Set initial infection at center
        self.grid[size//2, size//2] = 1
        
        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
    def count_states(self):
        """Count the number of individuals in each state"""
        return {
            'susceptible': np.sum(self.grid == 0),
            'infected': np.sum(self.grid == 1),
            'recovered': np.sum(self.grid == 2),
            'vaccinated': np.sum(self.grid == 3)
        }
    
    def calculate_infection_probability(self, i, j, neighbors_state):
        """Calculate infection probability based on multiple factors"""
        infected_neighbors = sum(1 for n in neighbors_state if n == 1)
        if infected_neighbors == 0:
            return 0
        
        # Base probability depends on number of infected neighbors
        base_prob = self.infection_rate * (infected_neighbors / len(neighbors_state))
        
        # Modify probability based on current state
        if self.grid[i,j] == 3:  # Vaccinated
            base_prob *= (1 - self.vaccine_protection)
        
        return min(base_prob, 1.0)  # Cap at 100%
    
    def calculate_recovery_probability(self, infection_time):
        """Calculate recovery probability based on infection duration"""
        # Probability increases with infection duration
        base_prob = self.recovery_rate
        duration_factor = min(infection_time / 10, 1)  # Max boost after 10 days
        return min(base_prob + (duration_factor * 0.2), 0.95)  # Cap at 95%
    
    def perform_vaccination(self):
        """Vaccinate susceptible individuals"""
        susceptible_coords = np.where(self.grid == 0)
        if len(susceptible_coords[0]) > 0:
            # Calculate how many to vaccinate
            num_to_vaccinate = min(self.daily_vaccination, len(susceptible_coords[0]))
            # Randomly select individuals to vaccinate
            indices = np.random.choice(len(susceptible_coords[0]), num_to_vaccinate, replace=False)
            for idx in indices:
                i, j = susceptible_coords[0][idx], susceptible_coords[1][idx]
                self.grid[i,j] = 3
                self.vaccination_time[i,j] = 0
    
    def update(self):
        """Update the simulation state"""
        new_grid = self.grid.copy()
        
        # Update time trackers
        self.recovery_time[self.grid == 2] += 1
        self.vaccination_time[self.grid == 3] += 1
        self.infection_duration[self.grid == 1] += 1
        
        # Check immunity expiration
        recovered_expired = (self.recovery_time >= self.immunity_period) & (self.grid == 2)
        vaccine_expired = (self.vaccination_time >= self.vaccine_immunity_period) & (self.grid == 3)
        new_grid[recovered_expired | vaccine_expired] = 0
        
        # Process each cell
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] in [0, 3]:  # Susceptible or Vaccinated
                    # Gather neighbor states
                    neighbors = []
                    for di in [-1,0,1]:
                        for dj in [-1,0,1]:
                            if 0 <= i+di < self.size and 0 <= j+dj < self.size:
                                neighbors.append(self.grid[i+di,j+dj])
                    
                    # Calculate and apply infection
                    if np.random.random() < self.calculate_infection_probability(i, j, neighbors):
                        new_grid[i,j] = 1
                        self.infection_duration[i,j] = 0
                
                elif self.grid[i,j] == 1:  # Infected
                    # Calculate recovery based on infection duration
                    if np.random.random() < self.calculate_recovery_probability(self.infection_duration[i,j]):
                        new_grid[i,j] = 2
                        self.recovery_time[i,j] = 0
                        self.infection_duration[i,j] = 0
        
        # Perform vaccination
        self.grid = new_grid
        self.perform_vaccination()
        
    def animate(self, num_steps=200):
        """Run the animation"""
        self.im = plt.imshow(self.grid, cmap='viridis')
        plt.colorbar(self.im, label='State (0:Susceptible, 1:Infected, 2:Recovered, 3:Vaccinated)')
        
        def update_fig(frame):
            self.update()
            self.im.set_array(self.grid)
            counts = self.count_states()
            plt.title(f'Day {frame}\nInfected: {counts["infected"]}, '
                     f'Susceptible: {counts["susceptible"]}, '
                     f'Recovered: {counts["recovered"]}, '
                     f'Vaccinated: {counts["vaccinated"]}')
            return [self.im]
        
        anim = animation.FuncAnimation(self.fig, update_fig, frames=num_steps,
                                     interval=100, blit=False)
        plt.show()

# Example usage and testing different vaccination rates
def test_vaccination_rates(rates_to_test):
    """Test different vaccination rates and return results"""
    results = {}
    for rate in rates_to_test:
        sim = VirusSimulation(daily_vaccination=rate)
        # Run simulation and track results
        # This is a simplified version - you might want to run multiple trials
        # and average the results
        initial_infected = np.sum(sim.grid == 1)
        sim.animate(200)
        final_infected = np.sum(sim.grid == 1)
        results[rate] = {'initial': initial_infected, 'final': final_infected}
    return results

# Example run
if __name__ == "__main__":
    # Regular simulation
    #sim = VirusSimulation()
    #sim.animate(200)
    
    # Test different vaccination rates
    rates = [50, 100, 150, 200]
    results = test_vaccination_rates(rates)
    print("\nVaccination Rate Testing Results:")
    for rate, result in results.items():
        print(f"Daily Vaccination Rate: {rate}")
        print(f"Final infected count: {result['final']}")

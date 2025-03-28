import numpy as np
import matplotlib.pyplot as plt

class DiseaseSpreadSimulation:
    def __init__(self, population=100000, initial_infected=1, vaccinated_percent=0.2, 
                 recovery_time_range=(1, 7), isolation_period=2):
        self.population = population
        self.vaccinated = int(population * vaccinated_percent)
        self.susceptible = population - initial_infected - self.vaccinated
        self.infected = initial_infected
        self.recovered = 0
        self.isolated = 0  # People in isolation who cannot infect others
        self.recovery_time_range = recovery_time_range
        self.isolation_period = isolation_period
        
        # Track recovery times for each infected person
        self.recovery_times = []
        self.isolation_times = []
        
        # Track daily states
        self.history = {
            'susceptible': [self.susceptible],
            'infected': [self.infected],
            'recovered': [self.recovered],
            'vaccinated': [self.vaccinated]
        }
    
    def simulate_day(self):
        # Handle recoveries
        recoveries = 0
        for i in range(len(self.recovery_times)):
            self.recovery_times[i] -= 1
        while self.recovery_times and self.recovery_times[0] <= 0:
            self.recovery_times.pop(0)
            recoveries += 1
        
        # Update counts for recoveries
        self.infected -= recoveries
        self.recovered += recoveries
        
        # Handle isolation
        for i in range(len(self.isolation_times)):
            self.isolation_times[i] -= 1
        while self.isolation_times and self.isolation_times[0] <= 0:
            self.isolation_times.pop(0)
            self.isolated -= 1
        
        # Calculate new infections
        new_infections = 0
        if self.isolated == 0:  # Only non-isolated infected can infect others
            new_infections = min(self.infected * 2, self.susceptible)
        
        # Update counts for new infections
        self.susceptible -= new_infections
        self.infected += new_infections
        self.isolated += new_infections
        
        # Assign recovery times and isolation times to new infections
        self.recovery_times.extend(np.random.randint(self.recovery_time_range[0], 
                                                     self.recovery_time_range[1] + 1, 
                                                     size=new_infections))
        self.isolation_times.extend([self.isolation_period] * new_infections)
        
        # Update history
        self.history['susceptible'].append(self.susceptible)
        self.history['infected'].append(self.infected)
        self.history['recovered'].append(self.recovered)
        self.history['vaccinated'].append(self.vaccinated)
    
    def run_simulation(self, days=100):
        for _ in range(days):
            self.simulate_day()
    
    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['susceptible'], label='Susceptible', color='blue')
        plt.plot(self.history['infected'], label='Infected', color='red')
        plt.plot(self.history['recovered'], label='Recovered', color='green')
        plt.plot(self.history['vaccinated'], label='Vaccinated', color='orange')
        
        plt.title('Disease Spread Over Time')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True)
        plt.show()

# Run the simulation
sim = DiseaseSpreadSimulation()
sim.run_simulation(days=100)
sim.plot_results()

import gym
import numpy as np
from gym import spaces

class SolarP2PEnv(gym.Env):
    def __init__(self, num_solar_owners=5, num_non_solar_owners=5, scenario='normal'):
        super(SolarP2PEnv, self).__init__()
        
        self.num_solar_owners = num_solar_owners
        self.num_non_solar_owners = num_non_solar_owners
        self.num_agents = num_solar_owners + num_non_solar_owners + 1  # +1 for the grid
        self.scenario = scenario
        
        # Define action space (amount of energy to trade/consume for each agent)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents,), dtype=np.float32)
        
        # Define observation space (energy production, consumption, prices, battery level, time of day)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_agents, 5), dtype=np.float32)
        
        self.solar_energy_used = 0
        self.grid_energy_used = 0
        
        self.reset()
    
    def reset(self):
        # Initialize state
        self.state = np.zeros((self.num_agents, 5))
        
        # Set initial energy production for solar owners
        self.state[:self.num_solar_owners, 0] = np.random.uniform(0, 1, self.num_solar_owners)
        
        # Set initial energy consumption for all owners
        self.state[:self.num_solar_owners + self.num_non_solar_owners, 1] = np.random.uniform(0, 1, self.num_solar_owners + self.num_non_solar_owners)
        
        # Set initial prices
        self.state[:, 2] = np.random.uniform(0.1, 0.5, self.num_agents)
        
        # Set initial battery levels for solar owners
        self.state[:self.num_solar_owners, 3] = np.random.uniform(0, 1, self.num_solar_owners)
        
        # Set initial time of day (0-24 hours)
        self.state[:, 4] = 0
        
        self.solar_energy_used = 0
        self.grid_energy_used = 0
        
        return self.state
    
    def step(self, action):
        # Apply actions (energy trading)
        energy_balance = self.state[:, 0] - self.state[:, 1] + action
        
        # Update battery levels for solar owners
        self.state[:self.num_solar_owners, 3] = np.clip(self.state[:self.num_solar_owners, 3] + energy_balance[:self.num_solar_owners], 0, 1)
        
        # Calculate rewards
        rewards = self._calculate_rewards(energy_balance)
        
        # Update energy usage counters
        self.solar_energy_used += np.sum(np.maximum(0, energy_balance[:self.num_solar_owners]))
        self.grid_energy_used += np.sum(np.maximum(0, -energy_balance[:-1]))
        
        # Update state
        self._update_state()
        
        # Check if episode is done (for simplicity, we'll say it's done after 24 hours)
        done = self.state[0, 4] >= 24
        
        return self.state, rewards, done, {
            "energy_balance": energy_balance,
            "solar_energy_used": self.solar_energy_used,
            "grid_energy_used": self.grid_energy_used
        }
    
    def _calculate_rewards(self, energy_balance):
        rewards = np.zeros(self.num_agents)
        
        # Solar owners get rewarded for selling excess energy and penalized for grid dependency
        solar_rewards = np.maximum(0, energy_balance[:self.num_solar_owners]) * self.state[:self.num_solar_owners, 2]
        grid_penalty = np.maximum(0, -energy_balance[:self.num_solar_owners]) * self.state[-1, 2] * 1.2  # 20% markup for grid energy
        rewards[:self.num_solar_owners] = solar_rewards - grid_penalty
        
        # Non-solar owners get penalized for buying energy
        rewards[self.num_solar_owners:-1] = -np.abs(energy_balance[self.num_solar_owners:-1]) * self.state[self.num_solar_owners:-1, 2]
        
        # Grid's reward is based on the overall energy balance
        rewards[-1] = np.sum(np.maximum(0, -energy_balance[:-1])) * self.state[-1, 2]
        
        return rewards
    
    def _update_state(self):
        # Update time of day
        self.state[:, 4] = (self.state[:, 4] + 1) % 24
        
        # Update energy production based on time of day and scenario
        time_factor = np.sin(np.pi * self.state[0, 4] / 12) ** 2
        if self.scenario == 'low_power':
            production_factor = 0.5
        elif self.scenario == 'high_power':
            production_factor = 1.5
        else:  # normal scenario
            production_factor = 1.0
        self.state[:self.num_solar_owners, 0] = np.random.uniform(0.5, 1, self.num_solar_owners) * time_factor * production_factor
        
        # Update energy consumption (higher during morning and evening)
        consumption_factor = 0.5 + 0.5 * (np.sin(np.pi * (self.state[0, 4] - 6) / 12) ** 2)
        self.state[:self.num_solar_owners + self.num_non_solar_owners, 1] = np.random.uniform(0.3, 0.7, self.num_solar_owners + self.num_non_solar_owners) * consumption_factor
        
        # Update prices (higher during peak demand and affected by scenario)
        price_factor = 1 + 0.5 * consumption_factor
        if self.scenario == 'low_power':
            price_factor *= 2.0
        elif self.scenario == 'high_power':
            price_factor *= 0.7
        self.state[:, 2] = self.state[:, 2] * price_factor + np.random.uniform(-0.2, 0.2, self.num_agents)
        self.state[:, 2] = np.clip(self.state[:, 2], 0.1, 25.0)  # Ensure prices are between $0.1 and $25

        # Ensure grid price is always slightly higher
        self.state[-1, 2] = min(max(self.state[:, 2]) * 1.1, 25.0)

    def set_scenario(self, scenario):
        self.scenario = scenario
        self.reset()  # Reset all values when the mode is changed


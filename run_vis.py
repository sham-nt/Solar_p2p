from env import SolarP2PEnv
from vis import visualize_environment

def main():
    # Create the environment without loading any RL models
    env = SolarP2PEnv(num_solar_owners=5, num_non_solar_owners=5)
    
    # Run the visualization
    print("Starting Solar P2P Energy Trading Simulation...")
    visualize_environment(env)
    print("Visualization closed.")

if __name__ == "__main__":
    main()
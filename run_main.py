import torch
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env import SolarP2PEnv
from groq import Groq  # Import Groq API
from tqdm import tqdm  # For progress bars

# Set up Groq API client
groq_client = Groq(api_key="your_groq_api_key")  # Replace with your Groq API key

# Define the environment
def make_env():
    env = SolarP2PEnv(num_solar_owners=5, num_non_solar_owners=5)
    return env

# Create a vectorized environment
vec_env = DummyVecEnv([make_env])

# Define the DQN model
print("Setting up DQN model...")
dqn_model = DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
    tensorboard_log="./dqn_solar_trading/"
)

# Define the PPO model
print("Setting up PPO model...")
ppo_model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_solar_trading/"
)

# Train the DQN model
print("Training DQN model...")
with tqdm(total=100000, desc="DQN Training Progress") as pbar:
    def dqn_callback(locals_, globals_):
        pbar.update(1)  # Update progress bar for each step
    dqn_model.learn(total_timesteps=100000, callback=dqn_callback, log_interval=10)

# Train the PPO model
print("Training PPO model...")
with tqdm(total=100000, desc="PPO Training Progress") as pbar:
    def ppo_callback(locals_, globals_):
        pbar.update(1)  # Update progress bar for each step
    ppo_model.learn(total_timesteps=100000, callback=ppo_callback, log_interval=10)

# Evaluate the models with a timeout
def evaluate_model_with_timeout(model, env, n_eval_episodes=10, max_steps_per_episode=1000):
    print(f"Evaluating {model.__class__.__name__} model...")
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Timeout after max_steps_per_episode
            if episode_length >= max_steps_per_episode:
                print(f"Episode {episode + 1} timed out after {max_steps_per_episode} steps.")
                done = True

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1} finished with reward: {episode_reward}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean reward: {mean_reward} ± {std_reward}")
    return mean_reward, std_reward

# Evaluate DQN with a timeout
dqn_mean_reward, dqn_std_reward = evaluate_model_with_timeout(dqn_model, vec_env, n_eval_episodes=3, max_steps_per_episode=1000)

# Evaluate PPO with a timeout
ppo_mean_reward, ppo_std_reward = evaluate_model_with_timeout(ppo_model, vec_env, n_eval_episodes=3, max_steps_per_episode=1000)

# Generate a performance report using Groq API
def generate_performance_report(model_name, mean_reward, std_reward, solar_energy_used, grid_energy_used):
    prompt = f"""
    The {model_name} model was trained on the Solar P2P Energy Trading environment. 
    The mean reward achieved was {mean_reward} with a standard deviation of {std_reward}.
    The total solar energy used was {solar_energy_used}, and the total grid energy used was {grid_energy_used}.
    Write a detailed performance report analyzing the results and suggesting improvements.
    """
    
    # Use Groq API to generate the report
    response = groq_client.completions.create(
        model="groq-llama",  # Replace with the appropriate Groq model name
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Get energy usage from the environment
solar_energy_used = env.solar_energy_used
grid_energy_used = env.grid_energy_used

# Generate reports
print("Generating performance reports...")
with tqdm(total=2, desc="Generating Reports") as pbar:
    dqn_report = generate_performance_report("DQN", dqn_mean_reward, dqn_std_reward, solar_energy_used, grid_energy_used)
    pbar.update(1)  # Update progress bar for DQN report
    ppo_report = generate_performance_report("PPO", ppo_mean_reward, ppo_std_reward, solar_energy_used, grid_energy_used)
    pbar.update(1)  # Update progress bar for PPO report

# Print reports
print("\nDQN Performance Report:")
print(dqn_report)
print("\nPPO Performance Report:")
print(ppo_report)

# Save reports to files
with open("dqn_performance_report.txt", "w") as f:
    f.write(dqn_report)

with open("ppo_performance_report.txt", "w") as f:
    f.write(ppo_report)

# Run the visualization
from vis import visualize_environment
print("\nRunning visualization...")
visualize_environment(env)
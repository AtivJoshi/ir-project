import numpy as np
import json
import os
from tqdm import tqdm

# Import the new classes
from linucb import LinUCBAgent
# Assuming Environment is defined in MAB.utils per blueprint Part C
from MAB.utils import Environment 

def train_agent():
    # Configuration
    DATA_PATH = "bandit_data_train.jsonl"
    OUTPUT_PATH = "linucb_training_history.json"
    N_ARMS = 5
    N_FEATURES = 5
    ALPHA = 0.1  # Exploration parameter
    TOTAL_STEPS = 50000  # Adjust based on dataset size

    print(f"Initializing Environment from {DATA_PATH}...")
    try:
        train_env = Environment(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please run Phase 1 (Data Generation) first.")
        return

    print(f"Initializing LinUCB Agent (Arms={N_ARMS}, Features={N_FEATURES}, Alpha={ALPHA})...")
    agent = LinUCBAgent(n_arms=N_ARMS, n_features=N_FEATURES, alpha=ALPHA)

    history = []
    cumulative_reward = 0.0

    print("Starting Training Loop...")
    # tqdm provides a progress bar
    for step in tqdm(range(TOTAL_STEPS), desc="Training"):
        # 1. Get Context
        # The Environment cycles through the pre-computed dataset
        query_data = train_env.get_next_query()
        
        # Ensure features are a numpy array
        context = np.array(query_data['features'])
        
        # 2. Select Action (Bandit Decision)
        chosen_arm = agent.select_arm(context)
        
        # 3. Get Reward (Simulate Partial Feedback)
        # We only reveal the reward for the arm we actually picked
        reward = train_env.get_reward(train_env.index, chosen_arm)
        
        # 4. Update Policy
        agent.update(chosen_arm, context, reward)
        
        # 5. Logging
        cumulative_reward += reward
        
        log_entry = {
            'step': step,
            'query_id': query_data.get('query_id', 'unknown'),
            'chosen_arm': int(chosen_arm),
            'reward': float(reward),
            'optimal_arm': int(query_data.get('optimal_arm', -1)),
            'regret': float(query_data['rewards'][query_data['optimal_arm']] - reward),
            'cumulative_reward': cumulative_reward
        }
        history.append(log_entry)

    # Save training history for analysis (Plotting Regret/Arm Distribution)
    print(f"Saving training history to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(history, f)
    
    print("Training Complete.")

if __name__ == "__main__":
    train_agent()
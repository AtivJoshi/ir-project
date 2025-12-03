import requests
import os
import json
import re
import argparse
import torch
import numpy as np
import random
import torch.nn.functional as F
import functools
import pandas as pd
import ast  
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class Environment(object):
    def __init__(self, arms, dataset, args=None, preding=False):
        self.arms = arms
        self.dataset = dataset
        self.preding = preding
        self.index = -1
        self.alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.skip_dataset = []
        self._update_state()
        

    def _update_state(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0
        
        while self.dataset[self.index]['dataset_name'] in self.skip_dataset and not self.preding:
            self.index += 1
            if self.index >= len(self.dataset):
                self.index = 0

        self.state = self.dataset[self.index]['text']
        
        # self.state = np.random.randint(0, self.arms)
    def _index_to_arm(self,index):
        if type(index) == np.ndarray:
            assert len(index) == 1
            index = index[0]
        return self.alpha_values[int(index)]
        
    def get_state(self):
        return self.state
        # return self.state

    def _get_reward(self, arm):

        query_data = self.dataset[self.index]
        rewards = query_data.get("rewards", [0.0] * 5)
        

        if arm < 0 or arm >= len(rewards):
            print(f"Warning: Arm {arm} out of range for rewards list of length {len(rewards)}")
            return 0.0
        
        return float(rewards[int(arm)])
                            
    def _get_recall(self,arm):
        raise NotImplementedError
        method = self._index_to_arm(arm)
        return self.dataset[self.index][method+'_eval']['recall']

    def choose_arm(self, arm):
        reward = self._get_reward(arm)
        self._update_state()
        return reward
    
    def __len__(self):
        return len(self.dataset)
    
    
import numpy as np
import torch
import functools

def rbf_kernel(x, y, sigma=1.0, device='cpu'):
    x = x.to(device)
    y = y.to(device)
    return torch.exp(-torch.linalg.norm(x - y)**2 / (2 * sigma**2))

class FastKernelUCBAgent:
    def __init__(self, n_arms, kernel_fn=rbf_kernel, alpha=1.0, lambda_reg=1.0, device='cpu'):
        self.n_arms = n_arms
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.kernel_fn = functools.partial(kernel_fn, device=self.device)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.X = [[] for _ in range(n_arms)]
        self.y = [[] for _ in range(n_arms)] 
        self.K_inv = [[] for _ in range(n_arms)] 

    def _kernel_vector(self, x, X_list):
        return torch.stack([self.kernel_fn(x, xi) for xi in X_list])

    def select_arm(self, context_vector):
        context_vector = context_vector.to(self.device).float()

        p = torch.full((self.n_arms,), float("-inf"), device=self.device, dtype=torch.float32)
        for arm in range(self.n_arms):
            if len(self.X[arm]) == 0:
                p[arm] = float("inf")
                continue

            X_arm = self.X[arm] 
            y_arm = torch.stack(self.y[arm]).flatten() 
            K_inv = self.K_inv[arm] 

            k_vec = self._kernel_vector(context_vector, X_arm) 

            # Use torch.matmul for matrix multiplication
            mean = torch.matmul(k_vec, torch.matmul(K_inv, y_arm))
            k_xx = self.kernel_fn(context_vector, context_vector) 

            var = k_xx - torch.matmul(k_vec, torch.matmul(K_inv, k_vec))
            var = torch.maximum(var, torch.tensor(1e-9, device=self.device)) 
            std = torch.sqrt(var)
            p[arm] = mean + self.alpha * std
        return torch.argmax(p).item() 

    def update(self, arm, x_new, reward):
        x_new = x_new.to(self.device).float()

        X_arm = self.X[arm]
        K_inv = self.K_inv[arm]

        X_arm.append(x_new)
        self.y[arm].append(torch.tensor(reward, dtype=torch.float32, device=self.device))

        if len(X_arm) == 1:
            k_xx = self.kernel_fn(x_new, x_new)
            # Initialize K_inv as a torch.Tensor
            self.K_inv[arm] = torch.tensor([[1.0 / (k_xx + self.lambda_reg)]], device=self.device, dtype=torch.float32)
            return

        k_vec = self._kernel_vector(x_new, X_arm[:-1]) # torch.Tensor
        k_xx = self.kernel_fn(x_new, x_new) # torch.Tensor

        c = k_xx + self.lambda_reg - torch.matmul(k_vec, torch.matmul(K_inv, k_vec))
        n_old = len(X_arm) - 1

        # Initialize K_inv_new as a torch.Tensor
        K_inv_new = torch.zeros((n_old + 1, n_old + 1), device=self.device, dtype=torch.float32)

        term1 = torch.matmul(K_inv, k_vec)
        K_inv_new[:n_old, :n_old] = K_inv + torch.outer(term1, term1) / c

        v = -term1 / c
        K_inv_new[:n_old, n_old] = v
        K_inv_new[n_old, :n_old] = v
        K_inv_new[n_old, n_old] = 1.0 / c

        self.K_inv[arm] = K_inv_new
        
def train_agent():
    # Configuration
    DATA_PATH = "/Users/arihantbarjatya/Documents/compsci 646/bandit_data_train.jsonl"
    OUTPUT_PATH = "/Users/arihantbarjatya/Documents/compsci 646/fastkernel_ucb_training_history.jsonl"
    N_ARMS = 5
    N_FEATURES = 5 
    ALPHA = 1.0  
    LAMBDA_REG = 1.0 
    TOTAL_STEPS = 50000  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Initializing Environment from {DATA_PATH}...")
    try:
        dataset = []
        with open(DATA_PATH, 'r') as f:
            for line in f:
                data_entry = json.loads(line)
                data_entry['dataset_name'] = "bandit_data_train"
                dataset.append(data_entry)
        train_env = Environment(arms=N_ARMS, dataset=dataset)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please run Phase 1 (Data Generation) first.")
        return

    print(f"Initializing FastKernelUCBAgent (Arms={N_ARMS}, Alpha={ALPHA}, Lambda={LAMBDA_REG}, Device={device})...")
    agent = FastKernelUCBAgent(n_arms=N_ARMS, alpha=ALPHA, lambda_reg=LAMBDA_REG, device=device)

    history = []
    cumulative_reward = 0.0

    print("Starting Training Loop...")
    for step in tqdm(range(TOTAL_STEPS), desc="Training"):
        query_data = train_env.dataset[train_env.index]
        context_np = np.array(query_data.get('features', np.random.rand(N_FEATURES)))
        context_tensor = torch.from_numpy(context_np).float().to(device)
        chosen_arm = agent.select_arm(context_tensor)
        reward = train_env.choose_arm(chosen_arm)
        agent.update(chosen_arm, context_tensor, reward)
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

    print(f"Saving training history to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for entry in history:
            f.write(json.dumps(entry) + '\n')

    print("Training Complete.")
    
    
if __name__ == "__main__":
    train_agent()

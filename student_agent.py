# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

def get_action(obs):
    global Q_table
    with open('q_table.pkl', 'rb') as f:
        Q_table = pickle.load(f)

    # state = state_to_key(obs)
    state = obs
    # 測試階段採用 greedy policy（若找不到狀態則隨機選擇）
    if state in Q_table:
        q_values = Q_table[state]
        best_action = int(np.argmax(q_values))
        return best_action
    else:
        return random.choice(range(6))
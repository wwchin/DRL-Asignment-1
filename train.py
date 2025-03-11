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

# Q-learning 超參數設定
alpha = 0.1      # 學習率
gamma = 0.99     # 折扣因子
epsilon = 0.2    # 探索率（訓練時使用）
num_episodes = 5000  # 訓練迴圈數

# Q-table 儲存每個 state 的各動作價值，key 為 state tuple，value 為長度6的 numpy 陣列
Q_table = {}

def state_to_key(state):
    # 此處 state 為 tuple，可直接作為 key
    return state

def train_q_learning():
    global Q_table
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        
        while not done:
            key = state_to_key(state)
            # 若尚未訓練過此 state 或隨機探索，則隨機選動作
            if random.random() < epsilon or key not in Q_table:
                action = random.choice(range(6))
            else:
                q_values = Q_table[key]
                action = int(np.argmax(q_values))
            
            next_state, reward, done, _ = env.step(action)
            next_key = state_to_key(next_state)
            
            # 若 state 尚未初始化，則給予初始 Q 值
            if key not in Q_table:
                Q_table[key] = np.zeros(6)
            if next_key not in Q_table:
                Q_table[next_key] = np.zeros(6)
            
            # Q-learning 更新公式
            best_next = np.max(Q_table[next_key])
            Q_table[key][action] += alpha * (reward + gamma * best_next - Q_table[key][action])
            
            state = next_state
        
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed.")

def get_action(obs):
    global Q_table
    state = state_to_key(obs)
    # 測試階段採用 greedy policy（若找不到狀態則隨機選擇）
    if state in Q_table:
        q_values = Q_table[state]
        best_action = int(np.argmax(q_values))
        return best_action
    else:
        return random.choice(range(6))

if __name__ == "__main__":
    print("訓練 Q-learning agent...")
    train_q_learning()
    # 將訓練後的 Q_table 儲存檔案，方便之後載入使用
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)
    print("訓練完成，Q_table 儲存到 q_table.pkl")
    
    # 可進行一次測試執行
    env = SimpleTaxiEnv(fuel_limit=5000)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    while not done:
        action = get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
    print(f"測試執行：共 {step_count} 步驟，總分 {total_reward}")
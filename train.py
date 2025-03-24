# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from tqdm import tqdm
import argparse
from simple_custom_taxi_env import SimpleTaxiEnv
from model import TabularModel, NNModel
import torch

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.


# Q-table 儲存每個 state 的各動作價值，key 為 state tuple，value 為長度6的 numpy 陣列
Q_table = {}

def state_to_key(obs):
    taxi_row, taxi_col, _,_ ,_,_, _,_, _,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    grid_size = obs[9] + 1  # destiantion B position
    relative_row = (obs[0]*10) // grid_size
    relative_col = (obs[1]*10) // grid_size
    # return (relative_row, relative_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)


def train_q_learning():
    global Q_table, gamma, alpha, num_episodes, epsilon_start, epsilon_end
    grid_size = random.randint(5, 10)
    env = SimpleTaxiEnv(fuel_limit=5000, grid_size=grid_size)
    total_total_reward = 0
    epsilon = epsilon_start
    print(epsilon, gamma)
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            key = state_to_key(state)
            # 若尚未訓練過此 state 或隨機探索，則隨機選動作
            if random.random() < epsilon or key not in Q_table:
                action = random.choice(range(6))
            else:
                q_values = Q_table[key]
                action = int(np.argmax(q_values))
            
            next_state, reward, done, _ = env.step(action)
            next_key = next_state
            total_reward += reward
            
            # 若 state 尚未初始化，則給予初始 Q 值
            if key not in Q_table:
                Q_table[key] = np.zeros(6)
            if next_key not in Q_table:
                Q_table[next_key] = np.zeros(6)
            
            # Q-learning 更新公式
            best_next = np.max(Q_table[next_key])
            Q_table[key][action] += alpha * (reward + gamma * best_next - Q_table[key][action])
            
            state = next_state
        
        epsilon *= gamma
        epsilon = max(epsilon, epsilon_end)
        total_total_reward += total_reward

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed. Average reward: {total_total_reward / 500}, epsilon: {epsilon}")
            total_total_reward = 0
            with open("q_table.pkl", "wb") as f:
                pickle.dump(Q_table, f)


def train_nn(model: NNModel, num_episodes, grid=5, max_steps=5000, gamma=0.99):
    goal = 0
    total_total_reward = 0
    action_counter = [0, 0, 0, 0, 0, 0]
    for episode in tqdm(range(num_episodes)):
        
        grid_size = random.randint(5, grid)
        env = SimpleTaxiEnv(fuel_limit=5000, grid_size=grid_size)

        obs, _ = env.reset()
        model.mem_init(obs)
        done = False
        trajectory = []
        total_reward = 0
        step = 0
         
        while (not done) and (step < max_steps):
            action = model.get_action(obs)
            action_counter[action] += 1
            next_obs, reward, done, _ = env.step(action)
            # TODO: reward shaping

            trajectory.append((obs, action, reward))
            total_reward += reward
            
            obs = next_obs     
            
            step += 1       
            
        # update policy
        if step < max_steps:
            # print(f"Episode {episode+1}/{num_episodes} action counter: {action_counter}")
            # print("step:", step)
            # print("update!! reward:", total_reward)
            goal = goal + 1
            
            G = 0
            for t in reversed(range(len(trajectory))):
                obs, action, reward = trajectory[t]
                G = gamma * G + reward
                model.update(obs, action, reward)

        total_total_reward += total_reward
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed. Average reward: {total_total_reward / 100}")
            print(f"goal: {goal}")
            print(f"action counter: {[x/100 for x in action_counter]}")
            action_counter = [0, 0, 0, 0, 0, 0]
            total_total_reward = 0
            goal = 0
            torch.save(model.policy.state_dict(), "nn_model.pth")

        # if (episode + 1) % 10 == 0:
        #     print(f"Episode {episode+1}/{num_episodes} action counter: {action_counter}")
    
    return model

def get_action(obs, model):
    if model == 'q_learning':
        global Q_table
        state = state_to_key(obs)
        # 測試階段採用 greedy policy（若找不到狀態則隨機選擇）
        if state in Q_table:
            q_values = Q_table[state]
            best_action = int(np.argmax(q_values))
            return best_action
        else:
            return random.choice(range(6))
    elif model == 'nn':
        return nnModel.get_action(obs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a agent for SimpleTaxiEnv.")
    parser.add_argument('--model', type=str, default='q_learning', help='Model type: q_learning or nn')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.999, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.025, help='Exploration rate')
    args = parser.parse_args()

    # Update hyperparameters with parsed arguments
    global num_episodes, alpha, gamma, epsilon_start, epsilon_end, nnModel
    num_episodes = args.episodes
    alpha = args.alpha
    gamma = args.gamma
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    if args.model == 'q_learning':
        print("訓練 Q-learning agent...")
        train_q_learning()
        # 將訓練後的 Q_table 儲存檔案，方便之後載入使用
        with open("q_table.pkl", "wb") as f:
            pickle.dump(Q_table, f)
        print("訓練完成，Q_table 儲存到 q_table.pkl")
    elif args.model == 'nn':
        print("訓練 NN agent...")
        nnModel = NNModel(lr=0.001)
        train_nn(nnModel, num_episodes, grid=5, max_steps=5000)
        train_nn(nnModel, num_episodes, grid=10, max_steps=500)
        # 將訓練後的 NN model 儲存檔案，方便之後載入使用
        torch.save(nnModel.policy.state_dict(), "nn_model.pth")
        print("訓練完成，NN model 儲存到 nn_model.pth")
    else:
        print("模型選項錯誤，請選擇 q_learning 或 nn")
    
    # 可進行一次測試執行
    env = SimpleTaxiEnv(fuel_limit=5000)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    while not done:
        action = get_action(obs, args.model)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
    print(f"測試執行：共 {step_count} 步驟，總分 {total_reward}")
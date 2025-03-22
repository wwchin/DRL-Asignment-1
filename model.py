import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

class TabularModel:
    def __init__(self, path=None, action_size=6):
        if path is None:
            self.Q_table = {}
        else:
            with open(path, 'rb') as f:
                self.Q_table = pickle.load(f)
        self.action_size = action_size

    
    def state_to_key(self, obs):
        taxi_row, taxi_col, _,_ ,_,_, _,_, _,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        grid_size = obs[9] + 1  # destiantion B position
        relative_row = (obs[0]*10) // grid_size
        relative_col = (obs[1]*10) // grid_size
        return (relative_row, relative_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

    def get_action(self, obs):
        key = self.state_to_key(obs)
        if key in self.Q_table:
            q_values = self.Q_table[key]
            best_action = int(np.argmax(q_values))
            return best_action
        else:
            return random.choice(range(6))

class NNModel:
    def __init__(self, action_size=6, lr=0.01, pretrained_path=None):
        """
        ✅ 使用 PyTorch 實作 Softmax Policy。
        - 存儲動作機率的 PyTorch 張量
        - 使用 **交叉熵損失（Cross-Entropy Loss）** 來學習
        - 使用 **Adam 優化器**
        """
        self.state_size = 10
        self.action_size = action_size

        # ✅ 初始化策略網路（Policy Network）
        self.policy = nn.Sequential(
            nn.Linear(self.state_size, 64),  # 輸入層 (state_size 維度)
            nn.ReLU(),  # 啟動函數
            nn.Linear(64, 64),  # 隱藏層
            nn.ReLU(),  # 啟動函數
            nn.Linear(64, action_size),  # 輸出層（對應 action_size 維度）
        )
        if pretrained_path is not None:
            self.policy.load_state_dict(torch.load(pretrained_path))

        # ✅ 使用 SGD 優化器
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)

        # ✅ 交叉熵損失函數
        # self.criterion = nn.CrossEntropyLoss()

    def mem_init(self, obs):
        taxi_row, taxi_col, s0_r,s0_c ,s1_r,s1_c, s2_r,s2_c, s3_r,s3_c, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        self.carrying_passenger = False
        self.grid_size = s3_r + 1  # destiantion B position
        self.stations = [(s0_r,s0_c), (s1_r,s1_c), (s2_r,s2_c), (s3_r,s3_c)]
        self.possible_passengers = self.stations.copy()
        self.possible_destinations = self.stations.copy()
        self.target = self.find_closest_station(taxi_row, taxi_col, self.possible_passengers)
        

    def find_closest_station(self, taxi_row, taxi_col, stations):
        target = None
        d_min = float('inf')
        for station in stations:
            _, _, d = self.get_relative_pos((taxi_row, taxi_col), station)
            if d < d_min:
                d_min = d
                target = station
        return target

    def get_relative_pos(self, p0, p1):
        # print("p0", p0)
        # print("p1", p1)
        r = (p1[0] - p0[0])
        c = (p1[1] - p0[1])
        return (r, c, abs(r) + abs(c))
    
    def obs_to_state(self, obs):
        taxi_row, taxi_col, s0_r,s0_c ,s1_r,s1_c, s2_r,s2_c, s3_r,s3_c, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        # print("taxi", taxi_row, taxi_col)
        
        if not self.carrying_passenger:
            self.target = self.find_closest_station(taxi_row, taxi_col, self.possible_passengers)
        else:
            self.target = self.find_closest_station(taxi_row, taxi_col, self.possible_destinations)
        relative_row, relative_col, target_d = self.get_relative_pos((taxi_row, taxi_col), self.target)
        # print("target", self.target)
        
        # close enough to target
        if target_d <= 1 and not passenger_look and not destination_look:
            if not self.carrying_passenger:
                # print("possible_passengers", self.possible_passengers)
                if self.target in self.possible_passengers:
                    self.possible_passengers.remove(self.target)
                self.target = self.find_closest_station(taxi_row, taxi_col, self.possible_passengers)
                relative_row, relative_col, target_d = self.get_relative_pos((taxi_row, taxi_col), self.target)
            else:
                if self.target in self.possible_destinations:
                    self.possible_destinations.remove(self.target)
                self.target = self.find_closest_station(taxi_row, taxi_col, self.possible_destinations)
                relative_row, relative_col, target_d = self.get_relative_pos((taxi_row, taxi_col), self.target)
            # update the target

        can_pickup = passenger_look and target_d == 0
        can_dropoff = self.carrying_passenger and destination_look and target_d == 0
        
            
        return (relative_row, relative_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, can_pickup, can_dropoff)
        # return (relative_row, relative_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

    def get_action(self, obs):
        """
        ✅ 根據 Softmax 機率分佈採樣動作。
        """
        
        state = self.obs_to_state(obs)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 轉換為張量
        assert state_tensor.shape == (1, self.state_size), f"state_tensor.shape: {state_tensor.shape}"

        action_probs = torch.softmax(self.policy(state_tensor), dim=-1)  # 計算 Softmax 動作機率
        # if (obs[0], obs[1]) not in self.stations:
        #     action_probs[0, 5] = 0
        action = torch.multinomial(action_probs, 1).item()  # 根據機率採樣動作

        # see if agent correctly picks up passenger
        if not self.carrying_passenger and obs[-2] and abs(state[0]) + abs(state[1]) == 0:
            if action == 4:
                self.carrying_passenger = True
                print(f"Pickup passenger at {obs[0], obs[1]}")
        if action == 5:
            if self.carrying_passenger:
                # set to current location
                self.possible_passengers = [(obs[0], obs[1])]
                print(f"Dropoff passenger at {obs[0], obs[1]}")
            self.carrying_passenger = False
        
        return action

    def update(self, state, action, reward):
        """
        ✅ 使用 **交叉熵損失** 更新策略。
        """
        self.optimizer.zero_grad()
        state = self.obs_to_state(state)

        # ✅ 轉換 state 與 action 為 PyTorch 張量
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # print("state_tensor", state_tensor)
        # print("action_tensor", action_tensor)

        # ✅ 計算 Softmax 動作機率
        action_probs = torch.softmax(self.policy(state_tensor), dim=-1)
        # print("action_probs", action_probs)

        # ✅ 計算交叉熵損失（負對數似然）
        loss = -torch.log(torch.clamp(action_probs[0, action], min=1e-8)) * reward
        # print("reward", reward)
        # print("loss", loss)

        # ✅ 反向傳播與梯度更新
        loss.backward()
        self.optimizer.step()
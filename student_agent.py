# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from tqdm import tqdm
from model import TabularModel, NNModel

tabularModel = TabularModel('q_table.pkl')
# nnModel = NNModel(pretrained_path='nn_model.pth')

def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    return tabularModel.get_action(obs)
    # return nnModel.get_action(obs)

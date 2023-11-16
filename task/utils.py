import math
import numpy as np

import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import math
from collections import namedtuple, deque
import pickle
from tabulate import tabulate

from typing import Optional, Tuple
import numpy as np
import gym
from gym import spaces
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Arrow
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        """
        初始化 QNetwork。
        :param state_size: 输入状态的大小。
        :param action_size: 可行动作的数量。
        :param hidden_sizes: 各隐藏层的大小（列表或元组）。
        """
        super(QNetwork, self).__init__()
        # 构建网络
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], action_size)
        
    def forward(self, state):
        """
        网络的前向传播。
        :param state: 输入状态。
        :return: 每个动作的 Q 值。
        """
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        
    def save(self, filename):
        """保存为pth文件"""
        torch.save(self.state_dict(), filename)
        
    def update(self, optimizer, experiences, gamma):
        states = torch.tensor([exp.state for exp in experiences], dtype=torch.float32)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.tensor([exp.next_state for exp in experiences], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32)

        # 获取当前状态的预测 Q 值
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 获取下一个状态的最大预测 Q 值
        max_next_q_values = self.forward(next_states).detach().max(1)[0]
        expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        # 计算损失
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    def print_model_parameters(self):
        for param_tensor in self.state_dict():
            print(param_tensor, "\n", self.state_dict()[param_tensor])
        


class QTable:
    def __init__(self, actions, filename=None):
        self.actions = actions
        if not filename is None:
            self.load(filename)
        else:
            self.q_table = {}

    def get_q_value(self, state, action):
        """ 从Q-table中获取Q值，如果不存在则初始化为0 """
        # 将连续状态转换为字符串键
        state_key = self.state_to_key(state)
        return self.q_table.get((state_key, action), 0.0)

    def set_q_value(self, state, action, value):
        """ 更新Q-table中的Q值 """
        # 将连续状态转换为字符串键
        state_key = self.state_to_key(state)
        self.q_table[(state_key, action)] = value
        
    def show_q_table(self, n):
        """ 使用tabulate显示Q-table的前n条记录，每行显示一个状态下的不同动作的Q-value值 """
        states = set(key[0] for key in self.q_table.keys())
        data = []

        for state in sorted(list(states))[:n]:
            row = [state]
            for action in range(len(self.actions)):
                row.append(round(self.get_q_value(state, action),4))
            data.append(row)

        headers = ["State"] + [f"{action}" for action in self.actions]
        print(tabulate(data, headers=headers, tablefmt="pretty"))

    def state_to_key(self, state):
        """ 将连续状态转换为字符串键 """
        return str(state)
    
    def update(self, experience, lr, gamma):
        state, action, reward, next_state, _ = experience

        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = (1 - lr) * current_q + lr * (reward + gamma * max_next_q)
        self.set_q_value(state, action, new_q)

    def save(self, filename):
        """ 保存 Q-table 到文件 filename.pkl """
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)
            print(f"Q-table saved to {filename}.")

    def load(self, filename):
        """ 从文件加载 Q-table """
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)
            print(f"Q-table loaded from {filename}.")
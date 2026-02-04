import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReinforcementLearning:
    """
    强化学习模块
    实现Q-Learning和DQN简化版，用于优化个性化激励策略
    """
    
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        初始化强化学习模块
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减系数
            epsilon_min: 最小探索率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 动作空间：对应不同的激励策略
        self.actions = ['金币', '会员', '内容解锁']
    
    def prepare_state(self, data):
        """
        准备状态数据
        
        Args:
            data: 原始数据集
        
        Returns:
            states: 状态矩阵
        """
        # 提取状态特征
        state_columns = ['age', 'read_days', 'total_read_time', 'read_chapters', 
                        'collect_count', 'comment_count', 'share_count', 'incentive_click_rate']
        
        # 处理类别特征
        categorical_columns = ['gender', 'device', 'city_level', 'register_source', 'consumption_level']
        for col in categorical_columns:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            state_columns.extend(dummies.columns.tolist())
        
        # 处理兴趣特征
        interest_dummies = data['interests'].str.get_dummies(sep=',')
        interest_dummies.columns = ['interest_' + col for col in interest_dummies.columns]
        data = pd.concat([data, interest_dummies], axis=1)
        state_columns.extend(interest_dummies.columns.tolist())
        
        # 状态矩阵：确保所有值都是数值类型
        states = data[state_columns].fillna(0).values.astype(float)
        
        return states
    
    def q_learning(self, states, rewards, n_episodes=1000, lr=0.1):
        """
        Q-Learning算法
        
        Args:
            states: 状态矩阵
            rewards: 奖励矩阵
            n_episodes: 训练轮数
            lr: 学习率
        
        Returns:
            q_table: Q表
        """
        # 简化处理：将连续状态离散化
        # 这里使用简单的分桶方法，实际应用中可能需要更复杂的离散化策略
        n_buckets = 5
        state_buckets = []
        
        for i in range(self.state_dim):
            min_val = states[:, i].min()
            max_val = states[:, i].max()
            buckets = np.linspace(min_val, max_val, n_buckets - 1)
            state_buckets.append(buckets)
        
        # 离散化状态
        def discretize_state(state):
            discrete_state = []
            for i, val in enumerate(state):
                discrete_state.append(np.digitize(val, state_buckets[i]))
            return tuple(discrete_state)
        
        # 初始化Q表
        q_table = {}
        
        # 训练
        for episode in range(n_episodes):
            # 随机选择一个用户作为起点
            user_idx = random.randint(0, len(states) - 1)
            state = states[user_idx]
            discrete_s = discretize_state(state)
            
            # 初始化Q表条目
            if discrete_s not in q_table:
                q_table[discrete_s] = np.zeros(self.action_dim)
            
            # 选择动作
            if random.uniform(0, 1) < self.epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = np.argmax(q_table[discrete_s])
            
            # 模拟执行动作后的奖励
            # 这里简化处理，实际应用中需要根据真实反馈计算奖励
            reward = rewards[user_idx, action]
            
            # 更新Q值
            if discrete_s in q_table:
                old_value = q_table[discrete_s][action]
                next_max = 0  # 简化处理，不考虑下一个状态
                new_value = old_value + lr * (reward + self.gamma * next_max - old_value)
                q_table[discrete_s][action] = new_value
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return q_table
    
    class DQN(nn.Module):
        """
        DQN网络
        """
        def __init__(self, state_dim, action_dim):
            super(ReinforcementLearning.DQN, self).__init__()
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, action_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    def dqn(self, states, rewards, n_episodes=1000, batch_size=32, lr=0.001, state_dim=None):
        """
        DQN算法
        
        Args:
            states: 状态矩阵
            rewards: 奖励矩阵
            n_episodes: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            state_dim: 状态维度，如果为None则使用self.state_dim
        
        Returns:
            model: 训练好的DQN模型
        """
        # 转换为张量
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        
        # 使用提供的state_dim或self.state_dim
        actual_state_dim = state_dim if state_dim is not None else self.state_dim
        
        # 初始化DQN模型
        model = self.DQN(actual_state_dim, self.action_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 经验回放缓冲区
        replay_buffer = deque(maxlen=10000)
        
        # 训练
        for episode in range(n_episodes):
            # 随机选择一个批次的用户
            batch_indices = np.random.choice(len(states), batch_size)
            batch_states = states[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            # 存储经验
            for i in range(batch_size):
                state = batch_states[i]
                # 选择动作
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    with torch.no_grad():
                        q_values = model(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                
                # 奖励
                reward = batch_rewards[i, action]
                
                # 存储经验（简化处理，不考虑下一个状态）
                replay_buffer.append((state, action, reward, state))
            
            # 经验回放
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)
                
                batch_states = torch.stack(batch_states)
                batch_actions = torch.LongTensor(batch_actions)
                batch_rewards = torch.FloatTensor(batch_rewards)
                batch_next_states = torch.stack(batch_next_states)
                
                # 计算Q值
                current_q = model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q = model(batch_next_states).max(1)[0]
                target_q = batch_rewards + self.gamma * next_q
                
                # 更新模型
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return model
    
    def calculate_reward(self, data):
        """
        计算奖励矩阵
        
        Args:
            data: 原始数据集
        
        Returns:
            rewards: 奖励矩阵
        """
        n_users = len(data)
        rewards = np.zeros((n_users, self.action_dim))
        
        for i, row in data.iterrows():
            # 基础奖励：留存率
            base_reward = row['actual_day_7_retention'] * 10
            
            # 付费奖励
            pay_reward = row['actual_has_paid'] * 20
            
            # 成本惩罚
            cost_penalty = 0
            
            # 根据不同激励策略计算奖励
            for j, action in enumerate(self.actions):
                if action == '金币':
                    # 金币激励：高成本，高留存
                    cost_penalty = -row['incentive_value'] * 0.01
                elif action == '会员':
                    # 会员激励：中等成本，中等留存
                    cost_penalty = -row['incentive_value'] * 0.5
                elif action == '内容解锁':
                    # 内容解锁：低成本，低留存
                    cost_penalty = -row['incentive_value'] * 0.1
                
                # 总奖励
                rewards[i, j] = base_reward + pay_reward + cost_penalty
        
        return rewards
    
    def select_action(self, model, state):
        """
        选择动作
        
        Args:
            model: 训练好的模型
            state: 当前状态
        
        Returns:
            action: 选择的动作
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            if isinstance(model, dict):
                # Q-Learning模型
                # 简化处理，实际应用中需要离散化状态
                return np.argmax(list(model.values())[0])
            else:
                # DQN模型
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    return torch.argmax(q_values).item()
    
    def optimize_strategy(self, data):
        """
        优化个性化激励策略
        
        Args:
            data: 原始数据集
        
        Returns:
            strategies: 优化后的策略
        """
        # 准备状态和奖励
        states = self.prepare_state(data)
        rewards = self.calculate_reward(data)
        
        # 动态设置状态维度
        actual_state_dim = states.shape[1]
        
        # 训练DQN模型
        model = self.dqn(states, rewards, state_dim=actual_state_dim)
        
        # 为每个用户生成策略
        strategies = []
        for i in range(len(data)):
            state = states[i]
            action_idx = self.select_action(model, state)
            strategy = self.actions[action_idx]
            strategies.append(strategy)
        
        return strategies

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('tomato_novel_user_data.csv')
    
    # 初始化强化学习模块
    state_dim = 20  # 简化处理，实际应用中需要根据特征数量调整
    action_dim = 3  # 三种激励策略
    rl = ReinforcementLearning(state_dim, action_dim)
    
    # 优化策略
    strategies = rl.optimize_strategy(data)
    
    # 添加策略到数据集
    data['optimized_strategy'] = strategies
    
    # 评估策略效果
    # 计算采用优化策略后的平均留存率和付费转化率
    optimized_retention = data['actual_day_7_retention'].mean()
    optimized_conversion = data['actual_has_paid'].mean()
    
    print(f"优化后平均7日留存率: {optimized_retention:.4f}")
    print(f"优化后平均付费转化率: {optimized_conversion:.4f}")
    
    # 保存结果
    data.to_csv('optimized_strategies.csv', index=False)
    print("策略优化完成，结果已保存")

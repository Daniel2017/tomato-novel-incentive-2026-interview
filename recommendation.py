import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RecommendationSystem:
    """
    推荐算法模块
    实现双塔模型，用于实现激励与内容流量协同
    """
    
    def __init__(self, user_dim, item_dim, embedding_dim=64):
        """
        初始化推荐系统模块
        
        Args:
            user_dim: 用户特征维度
            item_dim: 物品特征维度
            embedding_dim: 嵌入维度
        """
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedding_dim = embedding_dim
    
    def prepare_user_features(self, data):
        """
        准备用户特征
        
        Args:
            data: 原始数据集
        
        Returns:
            user_features: 用户特征矩阵
        """
        # 提取用户特征
        user_columns = ['age', 'read_days', 'total_read_time', 'read_chapters', 
                       'collect_count', 'comment_count', 'share_count', 'incentive_click_rate']
        
        # 处理类别特征
        categorical_columns = ['gender', 'device', 'city_level', 'register_source', 'consumption_level']
        for col in categorical_columns:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            user_columns.extend(dummies.columns.tolist())
        
        # 处理兴趣特征
        interest_dummies = data['interests'].str.get_dummies(sep=',')
        interest_dummies.columns = ['interest_' + col for col in interest_dummies.columns]
        data = pd.concat([data, interest_dummies], axis=1)
        user_columns.extend(interest_dummies.columns.tolist())
        
        # 用户特征矩阵：确保所有值都是数值类型
        user_features = data[user_columns].fillna(0).values.astype(float)
        
        return user_features
    
    def prepare_item_features(self, data):
        """
        准备物品（激励）特征
        
        Args:
            data: 原始数据集
        
        Returns:
            item_features: 物品特征矩阵
        """
        # 提取物品特征
        # 这里简化处理，实际应用中可能需要更复杂的特征工程
        item_features = []
        
        for _, row in data.iterrows():
            # 激励类型特征
            incentive_type = row['incentive_strategy']
            incentive_type_features = [1 if incentive_type == '金币' else 0, 
                                      1 if incentive_type == '会员' else 0, 
                                      1 if incentive_type == '内容解锁' else 0]
            
            # 激励强度特征
            incentive_value = row['incentive_value'] / 1000  # 归一化
            
            # 组合特征
            item_feature = incentive_type_features + [incentive_value]
            item_features.append(item_feature)
        
        return np.array(item_features)
    
    class TwinTowerModel(nn.Module):
        """
        双塔模型
        """
        def __init__(self, user_dim, item_dim, embedding_dim):
            super(RecommendationSystem.TwinTowerModel, self).__init__()
            # 用户塔
            self.user_tower = nn.Sequential(
                nn.Linear(user_dim, 128),
                nn.ReLU(),
                nn.Linear(128, embedding_dim),
                nn.ReLU()
            )
            
            # 物品塔
            self.item_tower = nn.Sequential(
                nn.Linear(item_dim, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_dim),
                nn.ReLU()
            )
        
        def forward(self, user_features, item_features):
            # 用户嵌入
            user_embedding = self.user_tower(user_features)
            # 物品嵌入
            item_embedding = self.item_tower(item_features)
            
            # 计算相似度（余弦相似度）
            similarity = nn.functional.cosine_similarity(user_embedding, item_embedding)
            
            return similarity
    
    def train_twin_tower(self, user_features, item_features, labels, epochs=100, batch_size=32, lr=0.001):
        """
        训练双塔模型
        
        Args:
            user_features: 用户特征矩阵
            item_features: 物品特征矩阵
            labels: 标签（0或1，表示是否匹配）
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        
        Returns:
            model: 训练好的双塔模型
        """
        # 转换为张量
        user_features = torch.FloatTensor(user_features)
        item_features = torch.FloatTensor(item_features)
        labels = torch.FloatTensor(labels)
        
        # 动态获取特征维度
        actual_user_dim = user_features.shape[1]
        actual_item_dim = item_features.shape[1]
        
        # 初始化模型
        model = self.TwinTowerModel(actual_user_dim, actual_item_dim, self.embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(user_features))
            user_features_shuffled = user_features[indices]
            item_features_shuffled = item_features[indices]
            labels_shuffled = labels[indices]
            
            # 批次训练
            for i in range(0, len(user_features), batch_size):
                # 获取批次数据
                batch_user = user_features_shuffled[i:i+batch_size]
                batch_item = item_features_shuffled[i:i+batch_size]
                batch_labels = labels_shuffled[i:i+batch_size]
                
                # 前向传播
                similarity = model(batch_user, batch_item)
                
                # 计算损失
                loss = criterion(similarity, batch_labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 打印损失
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        return model
    
    def predict_similarity(self, model, user_features, item_features):
        """
        预测用户和物品的相似度
        
        Args:
            model: 训练好的双塔模型
            user_features: 用户特征矩阵
            item_features: 物品特征矩阵
        
        Returns:
            similarities: 相似度矩阵
        """
        # 转换为张量
        user_features = torch.FloatTensor(user_features)
        item_features = torch.FloatTensor(item_features)
        
        # 预测相似度
        with torch.no_grad():
            similarities = model(user_features, item_features)
        
        return similarities.numpy()
    
    def recommend_incentives(self, model, user_features, incentive_candidates):
        """
        为用户推荐激励策略
        
        Args:
            model: 训练好的双塔模型
            user_features: 用户特征
            incentive_candidates: 激励候选列表
        
        Returns:
            recommended_incentive: 推荐的激励策略
        """
        # 计算用户与每个激励候选的相似度
        similarities = []
        for incentive in incentive_candidates:
            # 构建激励特征
            incentive_features = self._build_incentive_features(incentive)
            incentive_features = torch.FloatTensor(incentive_features).unsqueeze(0)
            user_features_tensor = torch.FloatTensor(user_features).unsqueeze(0)
            
            # 计算相似度
            with torch.no_grad():
                similarity = model(user_features_tensor, incentive_features).item()
            similarities.append(similarity)
        
        # 选择相似度最高的激励
        recommended_idx = np.argmax(similarities)
        recommended_incentive = incentive_candidates[recommended_idx]
        
        return recommended_incentive
    
    def _build_incentive_features(self, incentive):
        """
        构建激励特征
        
        Args:
            incentive: 激励策略
        
        Returns:
            features: 激励特征
        """
        # 激励类型特征
        incentive_type_features = [1 if incentive == '金币' else 0, 
                                  1 if incentive == '会员' else 0, 
                                  1 if incentive == '内容解锁' else 0]
        
        # 激励强度特征（默认值）
        incentive_value = 0.5  # 归一化值
        
        # 组合特征
        features = incentive_type_features + [incentive_value]
        
        return features
    
    def evaluate(self, model, user_features, item_features, labels):
        """
        评估模型性能
        
        Args:
            model: 训练好的双塔模型
            user_features: 用户特征矩阵
            item_features: 物品特征矩阵
            labels: 标签
        
        Returns:
            metrics: 评估指标
        """
        # 预测相似度
        similarities = self.predict_similarity(model, user_features, item_features)
        
        # 转换为二分类预测
        predictions = (similarities > 0.5).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def generate_labels(self, data):
        """
        生成标签
        
        Args:
            data: 原始数据集
        
        Returns:
            labels: 标签（1表示匹配，0表示不匹配）
        """
        # 简化处理，使用7日留存作为标签
        # 实际应用中可能需要更复杂的标签定义
        labels = data['actual_day_7_retention'].values
        
        return labels

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('tomato_novel_user_data.csv')
    
    # 初始化推荐系统模块
    user_dim = 20  # 简化处理，实际应用中需要根据特征数量调整
    item_dim = 4   # 3个激励类型特征 + 1个激励强度特征
    rs = RecommendationSystem(user_dim, item_dim)
    
    # 准备特征和标签
    user_features = rs.prepare_user_features(data)
    item_features = rs.prepare_item_features(data)
    labels = rs.generate_labels(data)
    
    # 训练模型
    model = rs.train_twin_tower(user_features, item_features, labels)
    
    # 评估模型
    metrics = rs.evaluate(model, user_features, item_features, labels)
    print("模型评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 为用户推荐激励策略
    incentive_candidates = ['金币', '会员', '内容解锁']
    user_idx = 0
    user_feature = user_features[user_idx]
    recommended_incentive = rs.recommend_incentives(model, user_feature, incentive_candidates)
    print(f"为用户 {user_idx} 推荐的激励策略: {recommended_incentive}")

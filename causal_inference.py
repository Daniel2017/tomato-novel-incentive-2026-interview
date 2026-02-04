import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

class CausalInference:
    """
    因果推断模块
    实现Uplift Model和PSM，用于补贴效果评估和用户筛选
    """
    
    def __init__(self):
        """
        初始化因果推断模块
        """
        pass
    
    def prepare_features(self, data):
        """
        准备特征数据
        
        Args:
            data: 原始数据集
        
        Returns:
            X: 特征矩阵
            y: 目标变量
            treatment: 处理变量
        """
        # 提取特征
        feature_columns = ['age', 'read_days', 'total_read_time', 'read_chapters', 
                          'collect_count', 'comment_count', 'share_count', 'incentive_click_rate']
        
        # 处理类别特征
        categorical_columns = ['gender', 'device', 'city_level', 'register_source', 'consumption_level']
        for col in categorical_columns:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            feature_columns.extend(dummies.columns.tolist())
        
        # 处理兴趣特征
        interest_dummies = data['interests'].str.get_dummies(sep=',')
        interest_dummies.columns = ['interest_' + col for col in interest_dummies.columns]
        data = pd.concat([data, interest_dummies], axis=1)
        feature_columns.extend(interest_dummies.columns.tolist())
        
        # 目标变量：7日留存
        y = data['actual_day_7_retention']
        
        # 处理变量：是否给予激励（这里简化处理，实际中可能需要更复杂的定义）
        treatment = 1 - (data['incentive_strategy'] == '无')
        
        # 特征矩阵
        X = data[feature_columns].fillna(0)
        
        return X, y, treatment
    
    def psm(self, X, treatment, y, k=5):
        """
        倾向得分匹配（Propensity Score Matching）
        
        Args:
            X: 特征矩阵
            treatment: 处理变量
            y: 目标变量
            k: 匹配的最近邻数量
        
        Returns:
            matched_data: 匹配后的数据集
        """
        # 训练倾向得分模型
        ps_model = LogisticRegression()
        ps_model.fit(X, treatment)
        
        # 预测倾向得分
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # 分离处理组和对照组
        treated_idx = treatment == 1
        control_idx = treatment == 0
        
        # 为处理组中的每个样本找到对照组中的最近邻
        treated_X = X[treated_idx]
        treated_y = y[treated_idx]
        treated_ps = propensity_scores[treated_idx]
        
        control_X = X[control_idx]
        control_y = y[control_idx]
        control_ps = propensity_scores[control_idx]
        
        # 使用最近邻算法进行匹配
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(control_ps.reshape(-1, 1))
        
        # 为每个处理组样本找到匹配的对照组样本
        distances, indices = nn.kneighbors(treated_ps.reshape(-1, 1))
        
        # 构建匹配后的数据集
        matched_treated = pd.DataFrame({
            'propensity_score': treated_ps,
            'treatment': 1,
            'outcome': treated_y
        })
        
        matched_control = pd.DataFrame({
            'propensity_score': np.concatenate([control_ps[idx] for idx in indices]),
            'treatment': 0,
            'outcome': np.concatenate([control_y.iloc[idx].values for idx in indices])
        })
        
        matched_data = pd.concat([matched_treated, matched_control], axis=0)
        
        return matched_data
    
    def calculate_ate(self, matched_data):
        """
        计算平均处理效应（Average Treatment Effect）
        
        Args:
            matched_data: 匹配后的数据集
        
        Returns:
            ate: 平均处理效应
        """
        treated_outcome = matched_data[matched_data['treatment'] == 1]['outcome'].mean()
        control_outcome = matched_data[matched_data['treatment'] == 0]['outcome'].mean()
        ate = treated_outcome - control_outcome
        
        return ate
    
    def uplift_model(self, X, y, treatment):
        """
        Uplift Model 实现（基于二分类器方法）
        
        Args:
            X: 特征矩阵
            y: 目标变量
            treatment: 处理变量
        
        Returns:
            model: 训练好的Uplift模型
            X_test: 测试特征
            y_test: 测试目标
            treatment_test: 测试处理变量
        """
        # 分割数据集
        X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
            X, y, treatment, test_size=0.3, random_state=42
        )
        
        # 训练两个模型：处理组和对照组
        treated_idx = treatment_train == 1
        control_idx = treatment_train == 0
        
        # 处理组模型
        treated_model = RandomForestClassifier(n_estimators=100, random_state=42)
        treated_model.fit(X_train[treated_idx], y_train[treated_idx])
        
        # 对照组模型
        control_model = RandomForestClassifier(n_estimators=100, random_state=42)
        control_model.fit(X_train[control_idx], y_train[control_idx])
        
        # 组合模型
        class UpliftModel:
            def __init__(self, treated_model, control_model):
                self.treated_model = treated_model
                self.control_model = control_model
            
            def predict_uplift(self, X):
                # 预测处理组和对照组的概率
                treated_prob = self.treated_model.predict_proba(X)[:, 1]
                control_prob = self.control_model.predict_proba(X)[:, 1]
                # 计算uplift
                uplift = treated_prob - control_prob
                return uplift
        
        model = UpliftModel(treated_model, control_model)
        
        return model, X_test, y_test, treatment_test
    
    def evaluate_uplift_model(self, model, X_test, y_test, treatment_test):
        """
        评估Uplift模型
        
        Args:
            model: 训练好的Uplift模型
            X_test: 测试特征
            y_test: 测试目标
            treatment_test: 测试处理变量
        
        Returns:
            auuc: Area Under Uplift Curve
        """
        # 预测uplift
        uplift = model.predict_uplift(X_test)
        
        # 构建评估数据集
        eval_data = pd.DataFrame({
            'uplift': uplift,
            'treatment': treatment_test,
            'outcome': y_test
        })
        
        # 按uplift排序
        eval_data = eval_data.sort_values('uplift', ascending=False)
        
        # 计算AUUC
        n_treated = eval_data['treatment'].sum()
        n_control = len(eval_data) - n_treated
        
        # 累积收益
        cumulative_gain = 0
        gains = []
        
        for i, row in eval_data.iterrows():
            if row['treatment'] == 1:
                cumulative_gain += row['outcome'] / n_treated
            else:
                cumulative_gain -= row['outcome'] / n_control
            gains.append(cumulative_gain)
        
        # 计算AUUC（简化版）
        auuc = np.trapz(gains, dx=1/len(gains))
        
        return auuc
    
    def select_sensitive_users(self, model, X, threshold=0.1):
        """
        选择对补贴敏感的用户
        
        Args:
            model: 训练好的Uplift模型
            X: 特征矩阵
            threshold: uplift阈值
        
        Returns:
            sensitive_users: 对补贴敏感的用户索引
        """
        # 预测uplift
        uplift = model.predict_uplift(X)
        
        # 选择uplift大于阈值的用户
        sensitive_users = np.where(uplift > threshold)[0]
        
        return sensitive_users

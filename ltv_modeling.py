import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LTVModeling:
    """
    LTV建模模块
    实现线性回归和特征交叉，用于预测用户长期价值，指导补贴策略
    """
    
    def __init__(self):
        """
        初始化LTV建模模块
        """
        pass
    
    def prepare_features(self, data):
        """
        准备特征数据
        
        Args:
            data: 原始数据集
        
        Returns:
            X: 特征矩阵
            y: 目标变量（LTV）
        """
        # 提取基础特征
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
        
        # 特征交叉
        # 这里简化处理，实际应用中可能需要更复杂的特征交叉
        cross_features = self._generate_cross_features(data)
        data = pd.concat([data, cross_features], axis=1)
        feature_columns.extend(cross_features.columns.tolist())
        
        # 构建目标变量（LTV）
        # 简化处理：基于7日留存和付费情况预测长期价值
        y = self._calculate_ltv(data)
        
        # 特征矩阵
        X = data[feature_columns].fillna(0)
        
        return X, y
    
    def _generate_cross_features(self, data):
        """
        生成特征交叉
        
        Args:
            data: 原始数据集
        
        Returns:
            cross_features: 特征交叉矩阵
        """
        cross_features = pd.DataFrame()
        
        # 活跃度与消费能力交叉
        cross_features['active_consumption'] = data['read_days'] * (data['consumption_level'].map({'低': 1, '中': 2, '高': 3}))
        
        # 年龄与兴趣交叉（简化处理）
        cross_features['age_interest_score'] = data['age'] * data['interests'].apply(lambda x: len(x.split(',')))
        
        # 激励点击率与留存交叉
        cross_features['click_retention'] = data['incentive_click_rate'] * data['actual_day_7_retention']
        
        return cross_features
    
    def _calculate_ltv(self, data):
        """
        计算LTV（简化版）
        
        Args:
            data: 原始数据集
        
        Returns:
            ltv: 用户长期价值
        """
        # 基础LTV
        base_ltv = 10  # 基础价值
        
        # 留存因子
        retention_factor = 1 + data['actual_day_7_retention'] * 5
        
        # 付费因子
        pay_factor = 1 + data['actual_has_paid'] * 10
        
        # 活跃度因子
        activity_factor = 1 + (data['read_days'] / 7) * 3
        
        # 消费能力因子
        consumption_map = {'低': 1, '中': 1.5, '高': 2}
        consumption_factor = data['consumption_level'].map(consumption_map)
        
        # 计算最终LTV
        ltv = base_ltv * retention_factor * pay_factor * activity_factor * consumption_factor
        
        return ltv
    
    def train_linear_regression(self, X, y):
        """
        训练线性回归模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            model: 训练好的线性回归模型
            metrics: 模型评估指标
        """
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred)
        
        return model, metrics
    
    def train_random_forest(self, X, y):
        """
        训练随机森林回归模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            model: 训练好的随机森林回归模型
            metrics: 模型评估指标
        """
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred)
        
        return model, metrics
    
    def evaluate_model(self, y_true, y_pred):
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            metrics: 评估指标
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def predict_ltv(self, model, X):
        """
        预测LTV
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
        
        Returns:
            ltv_pred: LTV预测值
        """
        ltv_pred = model.predict(X)
        return ltv_pred
    
    def segment_users_by_ltv(self, ltv_pred, thresholds=[20, 50]):
        """
        根据LTV预测值对用户进行分层
        
        Args:
            ltv_pred: LTV预测值
            thresholds: 分层阈值
        
        Returns:
            segments: 用户分层结果
        """
        segments = []
        for ltv in ltv_pred:
            if ltv < thresholds[0]:
                segments.append('低价值用户')
            elif ltv < thresholds[1]:
                segments.append('中价值用户')
            else:
                segments.append('高价值用户')
        
        return segments
    
    def generate_subsidy_strategy(self, segments):
        """
        根据用户分层生成补贴策略
        
        Args:
            segments: 用户分层结果
        
        Returns:
            strategies: 补贴策略
        """
        strategies = []
        for segment in segments:
            if segment == '高价值用户':
                # 高价值用户：高补贴，重点留存
                strategies.append('高价值补贴策略')
            elif segment == '中价值用户':
                # 中价值用户：中等补贴，平衡成本和留存
                strategies.append('中价值补贴策略')
            else:
                # 低价值用户：低补贴，控制成本
                strategies.append('低价值补贴策略')
        
        return strategies
    
    def optimize_subsidy_budget(self, ltv_pred, base_budget=100000):
        """
        优化补贴预算分配
        
        Args:
            ltv_pred: LTV预测值
            base_budget: 总预算
        
        Returns:
            budget_allocation: 预算分配方案
        """
        # 计算每个用户的预算分配权重
        weights = ltv_pred / ltv_pred.sum()
        
        # 分配预算
        budget_allocation = weights * base_budget
        
        return budget_allocation

if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('tomato_novel_user_data.csv')
    
    # 初始化LTV建模模块
    ltv_model = LTVModeling()
    
    # 准备特征和目标变量
    X, y = ltv_model.prepare_features(data)
    
    # 训练线性回归模型
    lr_model, lr_metrics = ltv_model.train_linear_regression(X, y)
    print("线性回归模型评估指标:")
    for key, value in lr_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 训练随机森林模型
    rf_model, rf_metrics = ltv_model.train_random_forest(X, y)
    print("\n随机森林模型评估指标:")
    for key, value in rf_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 预测LTV
    ltv_pred = ltv_model.predict_ltv(rf_model, X)
    
    # 用户分层
    segments = ltv_model.segment_users_by_ltv(ltv_pred)
    
    # 生成补贴策略
    strategies = ltv_model.generate_subsidy_strategy(segments)
    
    # 优化预算分配
    budget_allocation = ltv_model.optimize_subsidy_budget(ltv_pred)
    
    # 输出结果
    print(f"\n用户分层结果:")
    print(pd.Series(segments).value_counts())
    
    print(f"\n平均LTV预测值: {ltv_pred.mean():.2f}")
    print(f"总预算分配: {budget_allocation.sum():.2f}")

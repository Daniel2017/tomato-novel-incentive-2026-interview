import numpy as np
import pandas as pd
import sys

# 导入模块
sys.path.append('/Users/liuchunlei/Documents/trae_projects')
from data_generation import DataGenerator
from causal_inference import CausalInference
from reinforcement_learning import ReinforcementLearning
from recommendation import RecommendationSystem
from ltv_modeling import LTVModeling

class PersonalizedIncentiveSystem:
    """
    个性化激励补贴系统
    整合所有模块，实现完整的个性化激励补贴算法流程
    """
    
    def __init__(self):
        """
        初始化个性化激励系统
        """
        # 初始化各模块
        self.data_generator = DataGenerator()
        self.causal_inference = CausalInference()
        self.reinforcement_learning = None  # 后续初始化
        self.recommendation_system = None   # 后续初始化
        self.ltv_modeling = LTVModeling()
    
    def generate_data(self, num_users=10000):
        """
        生成数据
        
        Args:
            num_users: 生成的用户数量
        
        Returns:
            data: 生成的数据集
        """
        print(f"生成{num_users}个用户的数据...")
        data = self.data_generator.generate_full_dataset()
        print(f"数据生成完成，共包含{len(data)}条记录")
        return data
    
    def causal_analysis(self, data):
        """
        因果分析
        
        Args:
            data: 数据集
        
        Returns:
            uplift_model: 训练好的Uplift模型
            sensitive_users: 对补贴敏感的用户索引
        """
        print("进行因果分析...")
        
        # 准备特征
        X, y, treatment = self.causal_inference.prepare_features(data)
        
        # 倾向得分匹配
        matched_data = self.causal_inference.psm(X, treatment, y)
        ate = self.causal_inference.calculate_ate(matched_data)
        print(f"平均处理效应（ATE）: {ate:.4f}")
        
        # 训练Uplift模型
        uplift_model, X_test, y_test, treatment_test = self.causal_inference.uplift_model(X, y, treatment)
        
        # 评估Uplift模型
        auuc = self.causal_inference.evaluate_uplift_model(uplift_model, X_test, y_test, treatment_test)
        print(f"Uplift模型AUUC: {auuc:.4f}")
        
        # 选择对补贴敏感的用户
        sensitive_users = self.causal_inference.select_sensitive_users(uplift_model, X)
        print(f"对补贴敏感的用户数量: {len(sensitive_users)}")
        print(f"敏感用户占比: {len(sensitive_users)/len(X):.2%}")
        
        return uplift_model, sensitive_users
    
    def optimize_strategy(self, data):
        """
        优化激励策略
        
        Args:
            data: 数据集
        
        Returns:
            strategies: 优化后的策略
        """
        print("优化激励策略...")
        
        # 初始化强化学习模块
        state_dim = 20  # 简化处理，实际应用中需要根据特征数量调整
        action_dim = 3  # 三种激励策略
        self.reinforcement_learning = ReinforcementLearning(state_dim, action_dim)
        
        # 优化策略
        strategies = self.reinforcement_learning.optimize_strategy(data)
        
        # 添加策略到数据集
        data['optimized_strategy'] = strategies
        
        # 评估策略效果
        optimized_retention = data['actual_day_7_retention'].mean()
        optimized_conversion = data['actual_has_paid'].mean()
        
        print(f"优化后平均7日留存率: {optimized_retention:.4f}")
        print(f"优化后平均付费转化率: {optimized_conversion:.4f}")
        
        return strategies
    
    def recommend_incentives(self, data):
        """
        推荐激励策略
        
        Args:
            data: 数据集
        
        Returns:
            recommendations: 推荐的激励策略
        """
        print("推荐激励策略...")
        
        # 初始化推荐系统模块
        user_dim = 20  # 简化处理，实际应用中需要根据特征数量调整
        item_dim = 4   # 3个激励类型特征 + 1个激励强度特征
        self.recommendation_system = RecommendationSystem(user_dim, item_dim)
        
        # 准备特征和标签
        user_features = self.recommendation_system.prepare_user_features(data)
        item_features = self.recommendation_system.prepare_item_features(data)
        labels = self.recommendation_system.generate_labels(data)
        
        # 训练模型
        model = self.recommendation_system.train_twin_tower(user_features, item_features, labels)
        
        # 评估模型
        metrics = self.recommendation_system.evaluate(model, user_features, item_features, labels)
        print("推荐模型评估指标:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # 为每个用户推荐激励策略
        recommendations = []
        incentive_candidates = ['金币', '会员', '内容解锁']
        
        for i in range(len(data)):
            user_feature = user_features[i]
            recommendation = self.recommendation_system.recommend_incentives(model, user_feature, incentive_candidates)
            recommendations.append(recommendation)
        
        # 添加推荐到数据集
        data['recommended_incentive'] = recommendations
        
        return recommendations
    
    def ltv_analysis(self, data):
        """
        LTV分析
        
        Args:
            data: 数据集
        
        Returns:
            ltv_pred: LTV预测值
            segments: 用户分层结果
            strategies: 补贴策略
        """
        print("进行LTV分析...")
        
        # 准备特征和目标变量
        X, y = self.ltv_modeling.prepare_features(data)
        
        # 训练随机森林模型
        rf_model, rf_metrics = self.ltv_modeling.train_random_forest(X, y)
        print("LTV模型评估指标:")
        for key, value in rf_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # 预测LTV
        ltv_pred = self.ltv_modeling.predict_ltv(rf_model, X)
        
        # 用户分层
        segments = self.ltv_modeling.segment_users_by_ltv(ltv_pred)
        
        # 生成补贴策略
        strategies = self.ltv_modeling.generate_subsidy_strategy(segments)
        
        # 优化预算分配
        budget_allocation = self.ltv_modeling.optimize_subsidy_budget(ltv_pred)
        
        # 添加结果到数据集
        data['ltv_pred'] = ltv_pred
        data['user_segment'] = segments
        data['subsidy_strategy'] = strategies
        data['budget_allocation'] = budget_allocation
        
        # 输出结果
        print(f"用户分层结果:")
        print(pd.Series(segments).value_counts())
        
        print(f"平均LTV预测值: {ltv_pred.mean():.2f}")
        print(f"总预算分配: {budget_allocation.sum():.2f}")
        
        return ltv_pred, segments, strategies
    
    def integrate_strategies(self, data):
        """
        整合策略
        
        Args:
            data: 数据集
        
        Returns:
            final_strategies: 最终策略
        """
        print("整合策略...")
        
        # 简化处理：基于多个模型的结果，生成最终策略
        # 实际应用中可能需要更复杂的整合逻辑
        final_strategies = []
        
        for _, row in data.iterrows():
            # 优先考虑推荐系统的结果
            if 'recommended_incentive' in row:
                strategy = row['recommended_incentive']
            # 其次考虑强化学习的结果
            elif 'optimized_strategy' in row:
                strategy = row['optimized_strategy']
            # 最后考虑LTV分层的结果
            elif 'subsidy_strategy' in row:
                strategy = row['subsidy_strategy']
            else:
                # 默认策略
                strategy = '金币'
            
            final_strategies.append(strategy)
        
        # 添加最终策略到数据集
        data['final_strategy'] = final_strategies
        
        # 评估最终策略效果
        # 这里简化处理，实际应用中需要更详细的评估
        retention_rate = data['actual_day_7_retention'].mean()
        conversion_rate = data['actual_has_paid'].mean()
        
        print(f"最终策略效果:")
        print(f"7日留存率: {retention_rate:.4f}")
        print(f"付费转化率: {conversion_rate:.4f}")
        
        return final_strategies
    
    def run(self, num_users=10000):
        """
        运行完整流程
        
        Args:
            num_users: 生成的用户数量
        
        Returns:
            data: 处理后的数据集
        """
        print("开始运行个性化激励补贴系统...")
        
        # 1. 生成数据
        data = self.generate_data(num_users)
        
        # 2. 因果分析
        uplift_model, sensitive_users = self.causal_analysis(data)
        
        # 3. 优化策略
        strategies = self.optimize_strategy(data)
        
        # 4. 推荐激励
        recommendations = self.recommend_incentives(data)
        
        # 5. LTV分析
        ltv_pred, segments, ltv_strategies = self.ltv_analysis(data)
        
        # 6. 整合策略
        final_strategies = self.integrate_strategies(data)
        
        # 保存结果
        data.to_csv('personalized_incentive_results.csv', index=False)
        print("结果已保存到 personalized_incentive_results.csv")
        
        print("个性化激励补贴系统运行完成!")
        return data

if __name__ == '__main__':
    # 初始化系统
    system = PersonalizedIncentiveSystem()
    
    # 运行完整流程
    data = system.run(num_users=10000)
    
    # 打印最终结果
    print("\n最终结果摘要:")
    print(f"总用户数: {len(data)}")
    print(f"平均7日留存率: {data['actual_day_7_retention'].mean():.4f}")
    print(f"平均付费转化率: {data['actual_has_paid'].mean():.4f}")
    print(f"策略分布:")
    print(data['final_strategy'].value_counts())

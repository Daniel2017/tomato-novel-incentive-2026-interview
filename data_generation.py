import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

class DataGenerator:
    """
    番茄小说新用户数据生成器
    生成包含用户特征、行为数据和激励反馈的数据
    """
    
    def __init__(self, num_users=10000):
        """
        初始化数据生成器
        
        Args:
            num_users: 生成的用户数量
        """
        self.num_users = num_users
        self.user_ids = range(num_users)
        self.content_categories = ['都市', '言情', '玄幻', '科幻', '历史', '悬疑', '军事', '职场']
        self.incentive_types = ['金币', '会员', '内容解锁']
    
    def generate_user_features(self):
        """
        生成用户基础特征
        
        Returns:
            DataFrame: 用户基础特征数据
        """
        data = []
        
        for user_id in self.user_ids:
            # 基础特征
            age = random.randint(18, 50)
            gender = random.choice(['男', '女'])
            device = random.choice(['iOS', 'Android'])
            city_level = random.choice(['一线', '二线', '三线', '四线及以下'])
            
            # 注册来源
            register_source = random.choice(['应用商店', '广告投放', '社交分享', '自然搜索'])
            
            # 兴趣偏好
            interests = random.sample(self.content_categories, random.randint(1, 3))
            interest_str = ','.join(interests)
            
            # 消费能力
            consumption_level = random.choice(['低', '中', '高'])
            
            data.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'device': device,
                'city_level': city_level,
                'register_source': register_source,
                'interests': interest_str,
                'consumption_level': consumption_level
            })
        
        return pd.DataFrame(data)
    
    def generate_user_behavior(self, user_features):
        """
        生成用户行为数据
        
        Args:
            user_features: 用户基础特征数据
        
        Returns:
            DataFrame: 用户行为数据
        """
        data = []
        
        for _, user in user_features.iterrows():
            user_id = user['user_id']
            
            # 注册时间
            register_time = datetime.now() - timedelta(days=random.randint(0, 6))
            
            # 阅读行为
            read_days = random.randint(0, 7)
            total_read_time = read_days * random.randint(10, 120)  # 分钟
            read_chapters = read_days * random.randint(1, 10)
            
            # 互动行为
            collect_count = random.randint(0, 20)
            comment_count = random.randint(0, 10)
            share_count = random.randint(0, 5)
            
            # 激励相关行为
            incentive_click_rate = random.uniform(0.1, 0.8)
            
            # 付费行为
            has_paid = random.choice([0, 1]) if random.random() < 0.1 else 0
            if has_paid:
                first_pay_amount = random.randint(10, 100)
                first_pay_time = register_time + timedelta(hours=random.randint(1, 168))
            else:
                first_pay_amount = 0
                first_pay_time = None
            
            # 留存情况
            day_1_retention = 1 if random.random() < 0.6 else 0
            day_3_retention = 1 if day_1_retention and random.random() < 0.4 else 0
            day_7_retention = 1 if day_3_retention and random.random() < 0.25 else 0
            
            data.append({
                'user_id': user_id,
                'register_time': register_time,
                'read_days': read_days,
                'total_read_time': total_read_time,
                'read_chapters': read_chapters,
                'collect_count': collect_count,
                'comment_count': comment_count,
                'share_count': share_count,
                'incentive_click_rate': incentive_click_rate,
                'has_paid': has_paid,
                'first_pay_amount': first_pay_amount,
                'first_pay_time': first_pay_time,
                'day_1_retention': day_1_retention,
                'day_3_retention': day_3_retention,
                'day_7_retention': day_7_retention
            })
        
        return pd.DataFrame(data)
    
    def generate_incentive_data(self, user_features, user_behavior):
        """
        生成激励数据
        
        Args:
            user_features: 用户基础特征数据
            user_behavior: 用户行为数据
        
        Returns:
            DataFrame: 激励数据
        """
        data = []
        
        # 合并用户特征和行为数据
        user_data = pd.merge(user_features, user_behavior, on='user_id')
        
        for _, user in user_data.iterrows():
            user_id = user['user_id']
            
            # 激励策略：添加"无"激励策略的用户（20%概率）
            if random.random() < 0.2:
                incentive_strategy = '无'
                incentive_value = 0
                incentive_effect = 1.0  # 无激励效果
            else:
                incentive_strategy = random.choice(self.incentive_types)
                
                # 激励金额/强度
                if incentive_strategy == '金币':
                    incentive_value = random.randint(100, 1000)
                elif incentive_strategy == '会员':
                    incentive_value = random.randint(1, 7)  # 天数
                else:  # 内容解锁
                    incentive_value = random.randint(1, 5)  # 章节数
                
                # 激励效果（模拟）
                # 基于用户特征和行为模拟激励效果
                base_effect = 0.5
                
                # 年龄影响
                if user['age'] < 30:
                    age_factor = 1.2
                else:
                    age_factor = 0.8
                
                # 活跃度影响
                if user['read_days'] >= 3:
                    activity_factor = 1.3
                elif user['read_days'] >= 1:
                    activity_factor = 1.0
                else:
                    activity_factor = 0.7
                
                # 消费能力影响
                if user['consumption_level'] == '高':
                    consumption_factor = 1.4
                elif user['consumption_level'] == '中':
                    consumption_factor = 1.0
                else:
                    consumption_factor = 0.6
                
                # 计算最终激励效果
                incentive_effect = base_effect * age_factor * activity_factor * consumption_factor
                incentive_effect = min(max(incentive_effect, 0.1), 2.0)  # 限制在合理范围内
            
            # 实际留存和付费情况（考虑激励效果）
            actual_day_7_retention = 1 if random.random() < (user['day_7_retention'] * incentive_effect) else 0
            actual_has_paid = 1 if random.random() < (user['has_paid'] * incentive_effect * 1.5) else 0
            
            data.append({
                'user_id': user_id,
                'incentive_strategy': incentive_strategy,
                'incentive_value': incentive_value,
                'incentive_effect': incentive_effect,
                'actual_day_7_retention': actual_day_7_retention,
                'actual_has_paid': actual_has_paid
            })
        
        return pd.DataFrame(data)
    
    def generate_full_dataset(self):
        """
        生成完整数据集
        
        Returns:
            DataFrame: 完整数据集
        """
        # 生成用户基础特征
        user_features = self.generate_user_features()
        
        # 生成用户行为数据
        user_behavior = self.generate_user_behavior(user_features)
        
        # 生成激励数据
        incentive_data = self.generate_incentive_data(user_features, user_behavior)
        
        # 合并数据
        full_data = pd.merge(user_features, user_behavior, on='user_id')
        full_data = pd.merge(full_data, incentive_data, on='user_id')
        
        return full_data

if __name__ == '__main__':
    # 生成数据
    generator = DataGenerator(num_users=10000)
    dataset = generator.generate_full_dataset()
    
    # 保存数据
    dataset.to_csv('tomato_novel_user_data.csv', index=False)
    print(f"数据集生成完成，共包含 {len(dataset)} 条记录")
    print("数据预览:")
    print(dataset.head())

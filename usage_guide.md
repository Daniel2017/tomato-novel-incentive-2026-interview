# 项目使用指南

## 1. 项目概述

本项目是一个番茄小说新用户个性化激励补贴算法项目，专注于解决新用户留存低、补贴成本高、付费转化率低的业务痛点。通过整合因果推断、强化学习、推荐算法和LTV建模等核心技术，实现了个性化激励策略的自动优化。

## 2. 环境配置

### 2.1 依赖安装

项目使用Python 3.7+，需要安装以下依赖包：

```bash
# 创建虚拟环境（可选）
python3 -m venv venv

# 激活虚拟环境
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 安装依赖
pip install numpy pandas scikit-learn torch
```

### 2.2 文件结构

项目包含以下核心文件：

| 文件名 | 描述 |
|-------|-----|
| data_generation.py | 数据生成模块，生成符合业务场景的用户数据 |
| causal_inference.py | 因果推断模块，实现Uplift Model和PSM，评估补贴效果 |
| reinforcement_learning.py | 强化学习模块，实现Q-Learning和DQN，优化个性化激励策略 |
| recommendation.py | 推荐算法模块，实现双塔模型，实现激励与内容协同 |
| ltv_modeling.py | LTV建模模块，预测用户长期价值，指导补贴预算分配 |
| main.py | 主系统模块，整合所有功能，实现端到端流程 |
| project_documentation.md | 详细的项目文档 |
| exam_points_correspondence.md | 考点对应说明 |
| ai_tools_usage.md | AI工具运用说明 |
| interview_materials.md | 面试适配材料 |

## 3. 运行步骤

### 3.1 完整流程运行

最简单的运行方式是直接执行主脚本，它会自动运行完整的端到端流程：

```bash
# 运行完整流程
python main.py
```

执行后，系统会依次进行以下步骤：

1. **数据生成**：生成10000个用户的模拟数据
2. **因果分析**：使用Uplift Model和PSM评估补贴效果，筛选敏感用户
3. **策略优化**：使用强化学习（DQN）优化个性化激励策略
4. **激励推荐**：使用双塔模型实现激励与内容的协同推荐
5. **LTV分析**：预测用户长期价值，进行用户分层和预算优化
6. **策略整合**：整合多个模型的结果，生成最终的个性化激励策略
7. **结果保存**：将结果保存到`personalized_incentive_results.csv`文件

### 3.2 模块单独运行

如果需要单独运行某个模块，可以参考以下示例：

#### 3.2.1 数据生成

```python
from data_generation import DataGenerator

# 初始化数据生成器
generator = DataGenerator(num_users=10000)

# 生成数据
data = generator.generate_full_dataset()

# 保存数据
data.to_csv('user_data.csv', index=False)
print(f"数据生成完成，共包含{len(data)}条记录")
```

#### 3.2.2 因果分析

```python
import pandas as pd
from causal_inference import CausalInference

# 加载数据
data = pd.read_csv('user_data.csv')

# 初始化因果推断模块
ci = CausalInference()

# 准备特征
X, y, treatment = ci.prepare_features(data)

# 倾向得分匹配
matched_data = ci.psm(X, treatment, y)
ate = ci.calculate_ate(matched_data)
print(f"平均处理效应（ATE）: {ate:.4f}")

# 训练Uplift模型
uplift_model, X_test, y_test, treatment_test = ci.uplift_model(X, y, treatment)

# 评估Uplift模型
auuc = ci.evaluate_uplift_model(uplift_model, X_test, y_test, treatment_test)
print(f"Uplift模型AUUC: {auuc:.4f}")

# 选择对补贴敏感的用户
sensitive_users = ci.select_sensitive_users(uplift_model, X)
print(f"对补贴敏感的用户数量: {len(sensitive_users)}")
print(f"敏感用户占比: {len(sensitive_users)/len(X):.2%}")
```

#### 3.2.3 强化学习优化

```python
import pandas as pd
from reinforcement_learning import ReinforcementLearning

# 加载数据
data = pd.read_csv('user_data.csv')

# 初始化强化学习模块
state_dim = 20  # 简化处理，实际应用中需要根据特征数量调整
action_dim = 3  # 三种激励策略
rl = ReinforcementLearning(state_dim, action_dim)

# 优化策略
strategies = rl.optimize_strategy(data)

# 添加策略到数据集
data['optimized_strategy'] = strategies

# 评估策略效果
optimized_retention = data['actual_day_7_retention'].mean()
optimized_conversion = data['actual_has_paid'].mean()

print(f"优化后平均7日留存率: {optimized_retention:.4f}")
print(f"优化后平均付费转化率: {optimized_conversion:.4f}")
```

#### 3.2.4 推荐系统

```python
import pandas as pd
from recommendation import RecommendationSystem

# 加载数据
data = pd.read_csv('user_data.csv')

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
print("推荐模型评估指标:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# 为用户推荐激励策略
incentive_candidates = ['金币', '会员', '内容解锁']
user_idx = 0
user_feature = user_features[user_idx]
recommended_incentive = rs.recommend_incentives(model, user_feature, incentive_candidates)
print(f"为用户 {user_idx} 推荐的激励策略: {recommended_incentive}")
```

#### 3.2.5 LTV分析

```python
import pandas as pd
from ltv_modeling import LTVModeling

# 加载数据
data = pd.read_csv('user_data.csv')

# 初始化LTV建模模块
ltv_model = LTVModeling()

# 准备特征和目标变量
X, y = ltv_model.prepare_features(data)

# 训练随机森林模型
rf_model, rf_metrics = ltv_model.train_random_forest(X, y)
print("LTV模型评估指标:")
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
print(f"用户分层结果:")
print(pd.Series(segments).value_counts())

print(f"平均LTV预测值: {ltv_pred.mean():.2f}")
print(f"总预算分配: {budget_allocation.sum():.2f}")
```

## 4. 结果解读

### 4.1 输出文件

运行完整流程后，系统会生成以下输出文件：

| 文件名 | 描述 |
|-------|-----|
| personalized_incentive_results.csv | 包含所有用户的特征、行为、激励策略和预测结果的完整数据集 |

### 4.2 关键指标

在运行过程中，系统会输出以下关键指标：

| 指标 | 描述 | 预期效果 |
|-----|------|--------|
| 平均处理效应（ATE） | 补贴对用户留存的平均提升效果 | 正值，越大越好 |
| Uplift模型AUUC | Uplift模型的预测性能指标 | 越大越好，理想值>0.2 |
| 敏感用户占比 | 对补贴敏感的用户比例 | 通常在20%-40%之间 |
| 优化后平均7日留存率 | 应用优化策略后的7日留存率 | ≥8% |
| 优化后平均付费转化率 | 应用优化策略后的付费转化率 | ≥3% |
| 推荐模型评估指标 | 推荐模型的准确率、召回率等指标 | 准确率>80% |
| LTV模型评估指标 | LTV预测模型的性能指标 | R²>0.7 |
| 平均LTV预测值 | 用户长期价值的平均预测值 | 越大越好 |

### 4.3 结果分析

分析`personalized_incentive_results.csv`文件，可以得到以下洞察：

1. **用户分层分析**：根据LTV预测值，用户被分为高、中、低价值用户，不同层级的用户应该采用不同的激励策略。

2. **策略效果分析**：比较不同激励策略的效果，找出最适合不同用户群体的激励方式。

3. **预算分配分析**：根据LTV预测值和预算分配结果，优化补贴资源的分配，提高ROI。

4. **特征重要性分析**：分析哪些用户特征对留存和付费的影响最大，指导后续的特征工程和模型优化。

## 5. 面试应用

### 5.1 项目展示

在面试中展示项目时，建议按照以下步骤进行：

1. **项目背景**：简要介绍番茄小说的业务场景和面临的挑战。

2. **技术方案**：介绍使用的核心技术和算法，包括因果推断、强化学习、推荐算法和LTV建模。

3. **实现细节**：展示关键代码片段，说明技术实现的难点和解决方案。

4. **实验结果**：展示项目的实验结果和业务指标提升情况。

5. **学习收获**：分享通过项目获得的技术和业务 insights。

### 5.2 常见问题应对

面试中可能会遇到的问题及应对策略：

| 问题 | 应对策略 |
|-----|--------|
| 如何评估补贴效果 | 介绍Uplift Model和PSM的原理和应用，强调它们如何解决样本偏差问题 |
| 如何平衡探索与利用 | 介绍ε-greedy策略的原理和实现，说明如何根据业务场景调整探索率 |
| 如何处理高维稀疏特征 | 介绍特征工程的方法，包括独热编码、嵌入等技术 |
| 如何平衡留存提升和成本控制 | 介绍LTV建模的原理和应用，说明如何基于LTV进行预算分配 |
| 项目的核心亮点是什么 | 强调多模型融合的系统架构，以及对因果推断和强化学习等前沿算法的实际应用 |

## 6. 未来优化方向

1. **模型升级**：引入更先进的深度学习模型，如Transformer、Graph Neural Network等，提升预测精度。

2. **多目标优化**：设计更复杂的奖励函数，平衡留存、付费和成本多个目标。

3. **大模型应用**：探索大语言模型在用户兴趣理解和激励文案生成中的应用。

4. **实时决策**：优化模型推理速度，支持实时个性化激励决策。

5. **自动化运营**：实现策略自动调优，减少人工干预，提高运营效率。

## 7. 总结

本项目通过整合因果推断、强化学习、推荐算法和LTV建模等核心技术，实现了番茄小说新用户个性化激励补贴算法的完整解决方案。项目结构清晰，代码注释详细，易于理解和维护，具备实际部署的条件。

通过运行本项目，用户可以：

1. 了解字节激励增长算法岗的核心技术和业务考点
2. 掌握因果推断、强化学习、推荐算法和LTV建模等核心算法的实现
3. 学习如何将技术算法与业务目标相结合，实现业务价值
4. 为面试字节激励增长算法岗做好充分准备

希望本项目能够帮助用户快速掌握字节激励增长算法岗的核心技能，在面试中脱颖而出！
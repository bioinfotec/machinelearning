"""
决策树 - 是否去打篮球
====================
根据天气、温度、湿度、风等条件,决策是否出门打球
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font
from collections import Counter

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [天气(0=晴,1=阴,2=雨), 温度(℃), 湿度(%), 风速(km/h)]
X = np.array([
    [0, 28, 60, 10],  # 晴
    [0, 30, 65, 15],
    [0, 25, 70, 20],
    [0, 32, 55, 5],
    [1, 22, 80, 8],   # 阴
    [1, 25, 75, 12],
    [1, 28, 70, 10],
    [2, 20, 90, 25],  # 雨
    [2, 18, 95, 30],
    [2, 22, 85, 20],
    [2, 24, 80, 15],
    [0, 35, 50, 8],
    [1, 30, 65, 5],
    [2, 19, 88, 28]
])
# 标签: 0=不去, 1=去
y = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0])

feature_names = ['天气', '温度', '湿度', '风速']

# ============================================================
# 2. 手动实现决策树 (基于信息增益)
# ============================================================
def entropy(y):
    """计算信息熵"""
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

def information_gain(X, y, feature_idx, threshold):
    """计算信息增益"""
    parent_entropy = entropy(y)
    
    # 按阈值划分
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    
    # 加权平均子节点熵
    n = len(y)
    n_left, n_right = np.sum(left_mask), np.sum(right_mask)
    child_entropy = (n_left/n) * entropy(y[left_mask]) + (n_right/n) * entropy(y[right_mask])
    
    return parent_entropy - child_entropy

def find_best_split(X, y):
    """找到最佳划分"""
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_idx, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=4):
    """递归构建决策树"""
    # 终止条件
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2:
        most_common = Counter(y).most_common(1)[0][0]
        return {'leaf': True, 'prediction': most_common, 'samples': len(y)}
    
    # 找最佳划分
    feature, threshold, gain = find_best_split(X, y)
    
    if gain == 0:
        most_common = Counter(y).most_common(1)[0][0]
        return {'leaf': True, 'prediction': most_common, 'samples': len(y)}
    
    # 划分数据
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    # 递归构建子树
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    
    return {
        'leaf': False,
        'feature': feature,
        'threshold': threshold,
        'gain': gain,
        'left': left_tree,
        'right': right_tree
    }

def predict_tree(tree, x):
    """用决策树预测单个样本"""
    if tree['leaf']:
        return tree['prediction']
    
    if x[tree['feature']] <= tree['threshold']:
        return predict_tree(tree['left'], x)
    else:
        return predict_tree(tree['right'], x)

def print_tree(tree, indent=""):
    """打印决策树"""
    if tree['leaf']:
        label = "去" if tree['prediction'] == 1 else "不去"
        print(f"{indent}→ 预测: {label} (样本数={tree['samples']})")
    else:
        print(f"{indent}[{feature_names[tree['feature']]} ≤ {tree['threshold']}] (增益={tree['gain']:.3f})")
        print(f"{indent}  ├─ 是:", end=" ")
        print_tree(tree['left'], indent + "    ")
        print(f"{indent}  └─ 否:", end=" ")
        print_tree(tree['right'], indent + "    ")

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("决策树 - 是否去打篮球")
print("=" * 50)

tree = build_tree(X, y, max_depth=3)
print("\n决策树结构:")
print_tree(tree)

# ============================================================
# 4. 评估模型
# ============================================================
y_pred = np.array([predict_tree(tree, x) for x in X])
accuracy = np.mean(y_pred == y)
print(f"\n训练集准确率: {accuracy*100:.2f}%")

# ============================================================
# 5. 预测新情况
# ============================================================
X_new = np.array([
    [0, 27, 65, 10],  # 晴天,27度
    [2, 20, 90, 25],  # 雨天,20度
    [1, 25, 70, 15]   # 阴天,25度
])

print("\n新情况预测:")
for x in X_new:
    pred = predict_tree(tree, x)
    weather_name = ['晴', '阴', '雨'][int(x[0])]
    action = "去打球 ⚽" if pred == 1 else "不去 ❌"
    print(f"  天气={weather_name}, 温度={x[1]}℃, 湿度={x[2]}%, 风速={x[3]}km/h → {action}")

# ============================================================
# 6. 可视化 - 特征重要性
# ============================================================
# 计算每个特征的信息增益
gains = []
for feature_idx in range(X.shape[1]):
    _, _, gain = find_best_split(X, y)
    if find_best_split(X, y)[0] == feature_idx:
        gains.append(gain)
    else:
        gains.append(0)

# 简化: 重新计算每个特征的最佳增益
feature_gains = []
for feature_idx in range(X.shape[1]):
    best_gain = 0
    for threshold in np.unique(X[:, feature_idx]):
        gain = information_gain(X, y, feature_idx, threshold)
        best_gain = max(best_gain, gain)
    feature_gains.append(best_gain)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 特征重要性
axes[0].bar(feature_names, feature_gains, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0].set_ylabel('信息增益', fontsize=12)
axes[0].set_title('特征重要性', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')

# 图2: 决策路径示意
depths = [0, 1, 1, 2, 2]
labels = ['天气', '晴 → 温度', '阴/雨', '温度高 → 去', '温度低 → 不去']
y_pos = [4, 3, 3, 2, 2]
x_pos = [2, 1, 3, 0.5, 1.5]

for x, y, label in zip(x_pos, y_pos, labels):
    axes[1].text(x, y, label, ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'))

# 画连线
axes[1].plot([2, 1], [4, 3], 'k-', lw=1)
axes[1].plot([2, 3], [4, 3], 'k-', lw=1)
axes[1].plot([1, 0.5], [3, 2], 'k-', lw=1)
axes[1].plot([1, 1.5], [3, 2], 'k-', lw=1)

axes[1].set_xlim(-1, 5)
axes[1].set_ylim(1, 5)
axes[1].axis('off')
axes[1].set_title('决策路径示意', fontsize=14)

plt.tight_layout()
plt.show()

"""
KNN (K近邻) - 水果分类
======================
根据水果的重量和颜色深浅,预测水果类别
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [重量(g), 颜色深浅(0~1)]
X = np.array([
    [150, 0.2],  # 苹果
    [160, 0.3],
    [170, 0.25],
    [180, 0.3],
    [155, 0.35],
    [50,  0.9],  # 葡萄
    [60,  0.85],
    [55,  0.95],
    [65,  0.9],
    [70,  0.8],
    [200, 0.5],  # 橘子
    [210, 0.55],
    [220, 0.6],
    [190, 0.5],
    [230, 0.65]
])
# 标签: 0=苹果, 1=葡萄, 2=橘子
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

class_names = ['苹果', '葡萄', '橘子']
colors = ['red', 'purple', 'orange']

# ============================================================
# 2. 手动实现 KNN
# ============================================================
def knn_predict(X_train, y_train, X_test, k=3):
    """K近邻算法"""
    predictions = []
    
    for x in X_test:
        # 计算到所有训练样本的距离
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        
        # 找出最近的 k 个邻居
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        
        # 投票: 出现最多的类别为预测结果
        counts = np.bincount(k_labels)
        predictions.append(np.argmax(counts))
    
    return np.array(predictions)

# ============================================================
# 3. 训练和评估
# ============================================================
print("=" * 50)
print("KNN 分类器 - 水果识别")
print("=" * 50)

# 用训练集自身评估(简化演示)
k = 3
y_pred = knn_predict(X, y, X, k)
accuracy = np.mean(y_pred == y)
print(f"K = {k}")
print(f"训练集准确率: {accuracy*100:.2f}%")

# ============================================================
# 4. 预测新数据
# ============================================================
X_new = np.array([
    [165, 0.3],   # 看起来像苹果
    [58,  0.88],  # 看起来像葡萄
    [205, 0.55]   # 看起来像橘子
])

predictions = knn_predict(X, y, X_new, k=3)
print("\n新水果预测:")
for i, (features, pred) in enumerate(zip(X_new, predictions)):
    print(f"  重量={features[0]}g, 颜色深浅={features[1]} → {class_names[pred]}")

# ============================================================
# 5. 测试不同的 K 值
# ============================================================
print("\n不同 K 值的影响:")
for k_test in [1, 3, 5, 7]:
    y_pred_k = knn_predict(X, y, X, k_test)
    acc = np.mean(y_pred_k == y)
    print(f"  K={k_test}: 准确率={acc*100:.2f}%")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 数据分布
for cls, color, name in zip([0, 1, 2], colors, class_names):
    mask = y == cls
    axes[0].scatter(X[mask, 0], X[mask, 1], color=color, label=name, s=150, alpha=0.7, edgecolors='black')
axes[0].set_xlabel('重量 (g)', fontsize=12)
axes[0].set_ylabel('颜色深浅', fontsize=12)
axes[0].set_title('训练数据分布', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 决策边界
x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn_predict(X, y, grid_points, k=3).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])
for cls, color, name in zip([0, 1, 2], colors, class_names):
    mask = y == cls
    axes[1].scatter(X[mask, 0], X[mask, 1], color=color, label=name, s=150, alpha=0.9, edgecolors='black')

# 标记新预测点
for i, (point, pred) in enumerate(zip(X_new, predictions)):
    axes[1].scatter(point[0], point[1], color=colors[pred], s=300, 
                     marker='*', edgecolors='black', linewidths=2, label=f'新样本{i+1}' if i==0 else "")

axes[1].set_xlabel('重量 (g)', fontsize=12)
axes[1].set_ylabel('颜色深浅', fontsize=12)
axes[1].set_title(f'K={k} 决策边界 (★=待预测)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

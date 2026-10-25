"""
逻辑回归 - 是否购买商品
========================
预测: 用户是否会购买商品(0/1)
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [年龄, 月收入(千元)]
X = np.array([
    [22,  3],
    [25,  4],
    [28,  5],
    [30,  6],
    [35,  8],
    [40,  10],
    [45,  12],
    [50,  15],
    [55,  18],
    [60,  20],
    [23,  3.5],
    [33,  7],
    [42,  11],
    [48,  14],
    [58,  19]
])
# 标签: 是否购买(0=不买, 1=买)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])

# ============================================================
# 2. 手动实现逻辑回归
# ============================================================
def sigmoid(z):
    """Sigmoid 激活函数: 把任意值压缩到 0~1"""
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, n_iterations=10000):
    """
    使用梯度下降训练逻辑回归
    """
    # 数据归一化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    
    n, m = X_scaled.shape
    # 初始化参数: [b, w1, w2, ...]
    theta = np.zeros(m + 1)
    X_b = np.c_[np.ones((n, 1)), X_scaled]
    
    for i in range(n_iterations):
        # 前向传播
        z = X_b @ theta
        y_pred = sigmoid(z)
        
        # 计算梯度
        dw = (1/n) * X_b.T @ (y_pred - y)
        
        # 更新参数
        theta = theta - learning_rate * dw
    
    # 还原到原始尺度
    weights = theta[1:] / X_std
    bias = theta[0] - np.sum(weights * X_mean)
    
    return bias, weights

def predict(X, bias, weights, threshold=0.5):
    """预测类别"""
    z = X @ weights + bias
    prob = sigmoid(z)
    return (prob >= threshold).astype(int), prob

# ============================================================
# 3. 训练模型
# ============================================================
bias, weights = logistic_regression(X, y)

print("=" * 50)
print("逻辑回归 - 用户购买预测")
print("=" * 50)
print(f"截距 b  = {bias:.4f}")
print(f"权重 w1 (年龄)  = {weights[0]:.4f}")
print(f"权重 w2 (收入)  = {weights[1]:.4f}")

# ============================================================
# 4. 评估模型
# ============================================================
y_pred_class, y_pred_prob = predict(X, bias, weights)
accuracy = np.mean(y_pred_class == y)
print(f"\n准确率: {accuracy*100:.2f}%")

# ============================================================
# 5. 预测新用户
# ============================================================
X_new = np.array([
    [20,  2],
    [35,  10],
    [50,  15]
])

pred_class, pred_prob = predict(X_new, bias, weights)
print("\n新用户预测:")
for i, (features, cls, prob) in enumerate(zip(X_new, pred_class, pred_prob)):
    action = "购买" if cls == 1 else "不买"
    print(f"  年龄={features[0]}岁, 收入={features[1]}k → {action} (概率={prob:.2%})")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 数据分布
axes[0].scatter(X[y==0, 0], X[y==0, 1], color='red', label='不买', s=100, alpha=0.6, edgecolors='black')
axes[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', label='购买', s=100, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('年龄', fontsize=12)
axes[0].set_ylabel('月收入(千元)', fontsize=12)
axes[0].set_title('用户购买数据分布', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 决策边界
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-5, X[:,0].max()+5, 100),
                      np.linspace(X[:,1].min()-2, X[:,1].max()+2, 100))
Z = sigmoid(xx * weights[0] + yy * weights[1] + bias)

axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
axes[1].scatter(X[y==0, 0], X[y==0, 1], color='red', label='不买', s=100, edgecolors='black')
axes[1].scatter(X[y==1, 0], X[y==1, 1], color='blue', label='购买', s=100, edgecolors='black')
axes[1].set_xlabel('年龄', fontsize=12)
axes[1].set_ylabel('月收入(千元)', fontsize=12)
axes[1].set_title('决策边界 (黑色线=分界线)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""
多元线性回归 - 房价预测
========================
预测房价 = w1×面积 + w2×房间数 + w3×楼层 + b
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font
from mpl_toolkits.mplot3d import Axes3D

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [面积(平方米), 房间数, 楼层]
X = np.array([
    [50,  2, 5],
    [80,  3, 3],
    [100, 3, 7],
    [120, 4, 6],
    [150, 4, 10],
    [70,  2, 4],
    [90,  3, 8],
    [110, 3, 5],
    [130, 4, 9],
    [160, 5, 11]
])
# 目标: 房价(万元)
y = np.array([150, 220, 280, 320, 400, 180, 240, 290, 350, 450])

# ============================================================
# 2. 手动实现多元线性回归 - 正规方程
# ============================================================
def linear_regression_normal(X, y):
    """
    使用正规方程求解: θ = (X^T X)^(-1) X^T y
    """
    # 添加偏置列
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    # 正规方程求解
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta  # [b, w1, w2, w3]

# ============================================================
# 3. 训练模型
# ============================================================
theta = linear_regression_normal(X, y)

print("=" * 50)
print("多元线性回归 - 房价预测")
print("=" * 50)
print(f"截距 b  = {theta[0]:.4f}")
print(f"权重 w1 (面积)   = {theta[1]:.4f}")
print(f"权重 w2 (房间数) = {theta[2]:.4f}")
print(f"权重 w3 (楼层)   = {theta[3]:.4f}")
print(f"\n方程: 房价 = {theta[1]:.2f}×面积 + {theta[2]:.2f}×房间数 + {theta[3]:.2f}×楼层 + {theta[0]:.2f}")

# ============================================================
# 4. 预测新数据
# ============================================================
def predict(X_new, theta):
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
    return X_new_b @ theta

X_new = np.array([
    [85,  3, 6],
    [125, 4, 8],
    [140, 5, 9]
])

predictions = predict(X_new, theta)
print("\n预测结果:")
for i, (features, price) in enumerate(zip(X_new, predictions)):
    print(f"  面积={features[0]}m², 房间={features[1]}, 楼层={features[2]} → 房价={price:.2f}万元")

# ============================================================
# 5. 评估模型
# ============================================================
y_pred = predict(X, theta)
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
print(f"\n模型评估:")
print(f"  MSE (均方误差) = {mse:.2f}")
print(f"  R²  (决定系数) = {r2:.4f}")

# ============================================================
# 6. 可视化 - 实际值 vs 预测值
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 实际值 vs 预测值
axes[0].scatter(y, y_pred, color='blue', s=100, alpha=0.6, edgecolors='black')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0].set_xlabel('实际房价 (万元)', fontsize=12)
axes[0].set_ylabel('预测房价 (万元)', fontsize=12)
axes[0].set_title('实际值 vs 预测值', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 图2: 各特征的权重
features_name = ['面积', '房间数', '楼层']
axes[1].bar(features_name, theta[1:], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1].set_ylabel('权重值', fontsize=12)
axes[1].set_title('各特征对房价的影响', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

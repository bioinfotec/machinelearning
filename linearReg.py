import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
#                    手动实现线性回归
# ============================================================

# 1. 准备数据 (X是特征，y是目标值)
# 假设 X 是房屋面积 (平方米)，y 是房价 (万元)
X = np.array([50, 80, 100, 120, 150])
y = np.array([150, 200, 250, 280, 350])

# ============================================================
# 方法一：正规方程 (Normal Equation) - 解析解
# ============================================================
def normal_equation(X, y):
    """
    使用正规方程求解线性回归参数
    公式: theta = (X^T X)^(-1) X^T y
    """
    # 添加偏置项 (截距)
    X_b = np.c_[np.ones((len(X), 1)), X]  # 在X前面加一列1
    
    # 正规方程求解
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta[0], theta[1]  # 返回截距b and 权重w

# ============================================================
# 方法二：梯度下降 (Gradient Descent)
# ============================================================
def gradient_descent(X, y, learning_rate=0.01, n_iterations=10000):
    """
    使用梯度下降法求解线性回归参数
    """
    # 数据归一化 (关键步骤!)
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_scaled = (X - X_mean) / X_std
    
    # 初始化参数
    w = 0.0
    b = 0.0
    n = len(X_scaled)
    
    for i in range(n_iterations):
        # 计算预测值
        y_pred = w * X_scaled + b
        
        # 计算梯度
        dw = (1/n) * np.sum((y_pred - y) * X_scaled)  # w的梯度
        db = (1/n) * np.sum(y_pred - y)               # b的梯度
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    # 还原到原始尺度
    w_original = w / X_std
    b_original = b - w_original * X_mean
    
    return b_original, w_original

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("方法一：正规方程求解")
b1, w1 = normal_equation(X, y)
print(f"截距 (b): {b1:.4f}")
print(f"权重 (w): {w1:.4f}")
print(f"方程: y = {w1:.4f} * x + {b1:.4f}")

print("\n" + "=" * 50)
print("方法二：梯度下降求解")
b2, w2 = gradient_descent(X, y)
print(f"截距 (b): {b2:.4f}")
print(f"权重 (w): {w2:.4f}")
print(f"方程: y = {w2:.4f} * x + {b2:.4f}")

# ============================================================
# 4. 预测
# ============================================================
X_new = np.array([90, 130])

# 使用正规方程的结果预测
y_pred_normal = w1 * X_new + b1

print("\n" + "=" * 50)
print("预测结果:")
for area, price in zip(X_new, y_pred_normal):
    print(f"面积: {area} 平方米 -> 预测房价: {price:.2f} 万元")
    
# 使用梯度下降的结果预测
y_pred_gd = w2 * X_new + b2
print("\n" + "=" * 50)
print("使用梯度下降预测结果:")
for area, price in zip(X_new, y_pred_gd):
    print(f"面积: {area} 平方米 -> 预测房价: {price:.2f} 万元")

# ============================================================
# 5. 可视化
# ============================================================
plt.figure(figsize=(10, 6))

# 绘制原始数据点
plt.scatter(X, y, color='blue', s=100, label='原始数据', zorder=5)

# 绘制回归线
X_line = np.linspace(40, 160, 100)
y_line = w1 * X_line + b1
y_line_gd = w2 * X_line + b2

plt.plot(X_line, y_line, color='red', linewidth=2, 
         label=f'回归线: y = {w1:.4f}x + {b1:.4f}')

plt.plot(X_line, y_line_gd, color='orange', linewidth=2, linestyle='--',
         label=f'梯度下降线: y = {w2:.4f}x + {b2:.4f}')

# 绘制预测点
plt.scatter(X_new, y_pred_normal, color='green', s=150, marker='*', 
            label='预测点', zorder=5)
plt.scatter(X_new, y_pred_gd, color='purple', s=150, marker='X', label='梯度下降预测点', zorder=5)

plt.xlabel('房屋面积 (平方米)', fontsize=12)
plt.ylabel('房价 (万元)', fontsize=12)
plt.title('线性回归 - 房屋面积与房价关系', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

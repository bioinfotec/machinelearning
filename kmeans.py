"""
K-Means 聚类 - 客户分群
=======================
根据消费额和访问频次,将客户分为不同价值群体
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [年消费额(万元), 每月访问次数]
np.random.seed(42)
X = np.array([
    # 高价值客户群
    [8,  20], [9,  22], [10, 18], [8.5, 21],
    [9.5, 19], [11, 23], [10.5, 20],
    # 中等价值客户群
    [4,  10], [5,  12], [4.5, 11], [5.5, 13],
    [3.5, 9], [6, 14], [4.8, 11.5],
    # 低价值客户群
    [1,  3], [1.5, 2], [0.8, 4], [2, 3.5],
    [1.2, 2.5], [0.5, 3], [1.8, 4],
    # 离群点
    [12, 25], [0.3, 1]
])

# ============================================================
# 2. 手动实现 K-Means
# ============================================================
def kmeans(X, k=3, max_iters=100, random_state=42):
    """K-Means 聚类算法"""
    np.random.seed(random_state)
    
    # 随机选择 k 个中心点
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    history = {'centroids': [], 'labels': []}
    
    for iteration in range(max_iters):
        # 计算每个点到中心点的距离
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        
        # 分配每个点到最近的中心
        labels = np.argmin(distances, axis=0)
        
        # 记录历史
        history['centroids'].append(centroids.copy())
        history['labels'].append(labels.copy())
        
        # 更新中心点
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 检查收敛
        if np.allclose(centroids, new_centroids):
            print(f"  收敛于第 {iteration+1} 次迭代")
            break
        
        centroids = new_centroids
    
    return centroids, labels, history

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("K-Means 聚类 - 客户分群")
print("=" * 50)

k = 3
print(f"\n聚类数 K = {k}")
centroids, labels, history = kmeans(X, k=k)

print(f"\n聚类中心:")
for i, c in enumerate(centroids):
    print(f"  簇{i}: 年消费={c[0]:.2f}万元, 访问频次={c[1]:.1f}次/月")

cluster_names = ['低价值客户', '中等价值客户', '高价值客户']
# 按消费额排序: 找出哪个簇是低、中、高
sorted_indices = np.argsort(centroids[:, 0])

# ============================================================
# 4. 客户分群结果
# ============================================================
print("\n分群结果:")
for sorted_idx, original_idx in enumerate(sorted_indices):
    count = np.sum(labels == original_idx)
    print(f"  {cluster_names[sorted_idx]}: {count}人 (中心: 消费={centroids[original_idx, 0]:.1f}万元, 频次={centroids[original_idx, 1]:.1f}次)")

# ============================================================
# 5. 评估聚类效果 (惯性 - Inertia)
# ============================================================
def inertia(X, centroids, labels):
    """计算簇内平方和"""
    total = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        total += np.sum((cluster_points - centroids[i])**2)
    return total

print(f"\n聚类效果 (Inertia): {inertia(X, centroids, labels):.2f}")

# ============================================================
# 6. 肘部法则 - 选择最佳 K
# ============================================================
inertias = []
K_range = range(1, 7)
for k_test in K_range:
    c, l, _ = kmeans(X, k=k_test, max_iters=50)
    inertias.append(inertia(X, c, l))

print("\n不同 K 的 Inertia:")
for k_test, iner in zip(K_range, inertias):
    print(f"  K={k_test}: {iner:.2f}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 聚类结果
colors = ['red', 'gold', 'blue']
for i in range(k):
    mask = labels == i
    axes[0].scatter(X[mask, 0], X[mask, 1], color=colors[i], s=100, alpha=0.6, 
                     edgecolors='black', label=f'簇 {i+1}')
axes[0].scatter(centroids[:, 0], centroids[:, 1], color='black', s=300, marker='X', 
                 label='中心点', edgecolors='yellow', linewidths=2)
axes[0].set_xlabel('年消费额 (万元)', fontsize=12)
axes[0].set_ylabel('每月访问次数', fontsize=12)
axes[0].set_title('K-Means 聚类结果', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 迭代过程
for i, (cent, lab) in enumerate(zip(history['centroids'][:4], history['labels'][:4])):
    axes[1].scatter(X[:, 0], X[:, 1], c=lab, cmap='viridis', alpha=0.3, s=30)
    axes[1].scatter(cent[:, 0], cent[:, 1], color='red', s=100, marker='X', alpha=0.7)
    axes[1].set_title(f'第 {i+1} 次迭代', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    break  # 只画第一轮避免重叠

# 图3: 肘部法则
axes[2].plot(K_range, inertias, 'bo-', linewidth=2, markersize=10)
axes[2].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='最佳 K=3')
axes[2].set_xlabel('K 值', fontsize=12)
axes[2].set_ylabel('Inertia (簇内误差)', fontsize=12)
axes[2].set_title('肘部法则 - 选择 K', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

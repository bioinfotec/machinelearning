"""
SVM (支持向量机) - 二分类
=========================
找到一条最宽的分界线,将两类数据分开
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [重量(g), 长度(cm)]
X = np.array([
    # 类别 0: 小水果
    [80,  8],
    [90,  9],
    [100, 10],
    [85,  8.5],
    [95,  9.5],
    [110, 11],
    # 类别 1: 大水果
    [150, 15],
    [160, 16],
    [170, 17],
    [140, 14],
    [180, 18],
    [155, 15.5]
])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# ============================================================
# 2. 手动实现 SVM (线性, 简化版 SMO 算法)
# ============================================================
class SVM:
    """线性 SVM - 简化版"""
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 标签转为 -1 和 1
        y_ = np.where(y <= 0, -1, 1)
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 检查是否满足 KKT 条件
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # 正确分类,只正则化
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # 误分类,更新
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        
        return self
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.where(np.sign(approx) == -1, 0, 1)
    
    def decision_function(self, X):
        return np.dot(X, self.w) - self.b
    
    def get_support_vectors(self, X, threshold=0.1):
        """获取支持向量(距离分界线最近的点)"""
        distances = np.abs(self.decision_function(X))
        return X[distances < threshold + 1.0]

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("SVM 线性分类器 - 大小水果分类")
print("=" * 50)

svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

print(f"\n权重 w = {svm.w}")
print(f"偏置 b = {svm.b:.4f}")
print(f"分界方程: {svm.w[0]:.4f}×重量 + {svm.w[1]:.4f}×长度 = {svm.b:.4f}")

# ============================================================
# 4. 评估模型
# ============================================================
y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"\n训练集准确率: {accuracy*100:.2f}%")

# 找出支持向量
support_vectors = svm.get_support_vectors(X, threshold=0.1)
print(f"\n支持向量数量: {len(support_vectors)}")
print("支持向量 (决定分界线的关键点):")
for sv in support_vectors:
    print(f"  ({sv[0]}, {sv[1]})")

# ============================================================
# 5. 预测新数据
# ============================================================
X_new = np.array([
    [105, 10],   # 偏小
    [145, 14],   # 偏大
    [120, 12]    # 边界
])

preds = svm.predict(X_new)
print("\n新数据预测:")
for x, p in zip(X_new, preds):
    label = "大水果 🍎" if p == 1 else "小水果 🍒"
    margin = svm.decision_function([x])[0]
    print(f"  重量={x[0]}g, 长度={x[1]}cm → {label} (到分界线距离={margin:.3f})")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 图1: 数据 + 分界线
# 绘制分界线和间隔边界
xx = np.linspace(X[:, 0].min()-10, X[:, 0].max()+10, 100)
yy = np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100)
XX, YY = np.meshgrid(xx, yy)
Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

axes[0].contourf(XX, YY, Z, levels=[-10, -1, 0, 1, 10], 
                  colors=['#FFAAAA', '#FFDDDD', '#FFFFFF', '#DDDDFF', '#AAAAFF'], alpha=0.4)
axes[0].contour(XX, YY, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                 linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

# 绘制数据点
axes[0].scatter(X[y==0, 0], X[y==0, 1], color='red', label='小水果', s=150, edgecolors='black')
axes[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', label='大水果', s=150, edgecolors='black')

# 标记支持向量
if len(support_vectors) > 0:
    axes[0].scatter(support_vectors[:, 0], support_vectors[:, 1], 
                     facecolors='none', edgecolors='green', s=400, linewidths=2.5, 
                     label='支持向量')

axes[0].set_xlabel('重量 (g)', fontsize=12)
axes[0].set_ylabel('长度 (cm)', fontsize=12)
axes[0].set_title('SVM 分类结果 (黑色=分界线)', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 间隔示意图
axes[1].bar(['间隔宽度'], [2/np.linalg.norm(svm.w)], color='green', alpha=0.7, width=0.5)
axes[1].set_ylabel('宽度', fontsize=12)
axes[1].set_title(f'分类间隔宽度 = 2/||w|| = {2/np.linalg.norm(svm.w):.4f}', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, max(3, 3/np.linalg.norm(svm.w)))

plt.tight_layout()
plt.show()

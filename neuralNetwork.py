"""
神经网络 - XOR 问题
===================
最简单的神经网络,解决非线性可分问题
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据 - XOR 问题
# ============================================================
# 输入: [x1, x2]
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# 输出: 异或结果 (相同=0, 不同=1)
y = np.array([0, 1, 1, 0]).reshape(-1, 1)

# ============================================================
# 2. 手动实现神经网络
# ============================================================
def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Sigmoid 导数"""
    s = sigmoid(z)
    return s * (1 - s)

class NeuralNetwork:
    """简单的两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # 初始化权重
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]
        
        # 计算输出层误差
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 计算隐藏层误差
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def train(self, X, y, epochs=10000):
        """训练"""
        losses = []
        for i in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward(X, y)
            
            if i % 2000 == 0:
                print(f"  Epoch {i:5d}: Loss = {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return (output >= 0.5).astype(int), output

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("神经网络 - 解决 XOR 问题")
print("=" * 50)
print("\n为什么 XOR 需要神经网络?")
print("XOR 是非线性可分,单层感知机无法解决")
print("需要至少一个隐藏层才能画曲线分界线")

# 训练
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
print("\n开始训练:")
losses = nn.train(X, y, epochs=10001)

# ============================================================
# 4. 评估模型
# ============================================================
y_pred_class, y_pred_prob = nn.predict(X)
print("\n预测结果:")
print(f"{'输入':<10} {'真实':<8} {'预测概率':<12} {'预测类别':<8}")
print("-" * 45)
for i in range(len(X)):
    print(f"{str(X[i]):<10} {y[i,0]:<8} {y_pred_prob[i,0]:.4f}      {y_pred_class[i,0]:<8}")

accuracy = np.mean(y_pred_class == y)
print(f"\n准确率: {accuracy*100:.2f}%")

# ============================================================
# 5. 可视化
# ============================================================
fig = plt.figure(figsize=(16, 5))

# 图1: 损失曲线
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(losses, color='blue', linewidth=2)
ax1.set_xlabel('迭代次数', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('训练损失曲线', fontsize=14)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 图2: XOR 数据 + 决策边界
ax2 = fig.add_subplot(1, 3, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z, _ = nn.predict(grid)
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
for i in range(len(X)):
    if y[i, 0] == 1:
        ax2.scatter(X[i, 0], X[i, 1], color='blue', s=300, edgecolors='black', 
                     linewidths=2, label='1' if i==1 else "")
    else:
        ax2.scatter(X[i, 0], X[i, 1], color='red', s=300, edgecolors='black',
                     linewidths=2, label='0' if i==0 else "")

ax2.set_xlabel('x1', fontsize=12)
ax2.set_ylabel('x2', fontsize=12)
ax2.set_title('XOR 决策边界', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 网络结构图
ax3 = fig.add_subplot(1, 3, 3)
ax3.axis('off')

# 绘制神经元
input_pos = [(0.2, 0.8), (0.2, 0.3)]
hidden_pos = [(0.5, 0.95), (0.5, 0.65), (0.5, 0.35), (0.5, 0.05)]
output_pos = [(0.85, 0.5)]

# 输入层标签
for i, (x, y) in enumerate(input_pos):
    circle = plt.Circle((x, y), 0.05, color='lightblue', ec='black', lw=2)
    ax3.add_patch(circle)
    ax3.text(x-0.1, y, f'x{i+1}', ha='right', va='center', fontsize=12, fontweight='bold')

# 隐藏层
for i, (x, y) in enumerate(hidden_pos):
    circle = plt.Circle((x, y), 0.05, color='lightgreen', ec='black', lw=2)
    ax3.add_patch(circle)
    ax3.text(x, y, f'h{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

# 输出层
for i, (x, y) in enumerate(output_pos):
    circle = plt.Circle((x, y), 0.05, color='lightyellow', ec='black', lw=2)
    ax3.add_patch(circle)
    ax3.text(x+0.1, y, 'y', ha='left', va='center', fontsize=12, fontweight='bold')

# 绘制连接线
for x1, y1 in input_pos:
    for x2, y2 in hidden_pos:
        ax3.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.5)

for x1, y1 in hidden_pos:
    for x2, y2 in output_pos:
        ax3.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.5)

ax3.text(0.2, 1.0, '输入层', ha='center', fontsize=13, fontweight='bold')
ax3.text(0.5, 1.05, '隐藏层 (4个神经元)', ha='center', fontsize=13, fontweight='bold')
ax3.text(0.85, 0.7, '输出层', ha='center', fontsize=13, fontweight='bold')
ax3.set_xlim(-0.05, 1.0)
ax3.set_ylim(-0.05, 1.1)
ax3.set_title('神经网络结构', fontsize=14)

plt.tight_layout()
plt.show()

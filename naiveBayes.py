"""
朴素贝叶斯 - 垃圾邮件分类
========================
根据邮件中是否包含特定关键词,判断是否为垃圾邮件
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_chinese_font

# 统一配置中文字体,避免图表中文显示为方框
setup_chinese_font()

# ============================================================
# 1. 准备数据
# ============================================================
# 特征: [包含"免费"?, 包含"紧急"?, 包含"中奖"?, 长度超过100字?]
#        0=否, 1=是
X = np.array([
    # 垃圾邮件
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 0, 0, 1],
    # 正常邮件
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 1]
])
# 标签: 0=正常, 1=垃圾
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

feature_names = ['含"免费"', '含"紧急"', '含"中奖"', '长度>100字']

# ============================================================
# 2. 手动实现朴素贝叶斯
# ============================================================
class NaiveBayes:
    """朴素贝叶斯分类器"""
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 计算先验概率 P(y)
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.priors[i] = np.sum(y == c) / n_samples
        
        # 计算条件概率 P(x|y)
        self.likelihoods = np.zeros((n_classes, n_features, 2))  # 每个特征取 0/1
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            for j in range(n_features):
                # Laplace 平滑避免概率为0
                self.likelihoods[i, j, 0] = (np.sum(X_c[:, j] == 0) + 1) / (len(X_c) + 2)
                self.likelihoods[i, j, 1] = (np.sum(X_c[:, j] == 1) + 1) / (len(X_c) + 2)
        
        return self
    
    def predict(self, X):
        predictions = []
        posteriors_all = []
        
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                # 后验概率 ∝ 先验 × 似然
                log_prior = np.log(self.priors[i])
                log_likelihood = 0
                for j, val in enumerate(x):
                    log_likelihood += np.log(self.likelihoods[i, j, val])
                
                posteriors.append(log_prior + log_likelihood)
            
            posteriors_all.append(posteriors)
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions), np.array(posteriors_all)

# ============================================================
# 3. 训练模型
# ============================================================
print("=" * 50)
print("朴素贝叶斯 - 垃圾邮件分类")
print("=" * 50)

nb = NaiveBayes()
nb.fit(X, y)

print("\n先验概率:")
for i, c in enumerate(nb.classes):
    label = "垃圾邮件" if c == 1 else "正常邮件"
    print(f"  P({label}) = {nb.priors[i]:.4f}")

print("\n条件概率 (似然):")
for i, c in enumerate(nb.classes):
    label = "垃圾邮件" if c == 1 else "正常邮件"
    print(f"\n  [{label}]")
    for j, fname in enumerate(feature_names):
        p0 = nb.likelihoods[i, j, 0]
        p1 = nb.likelihoods[i, j, 1]
        print(f"    P({fname}=否 | {label}) = {p0:.4f}, P({fname}=是 | {label}) = {p1:.4f}")

# ============================================================
# 4. 评估模型
# ============================================================
y_pred, _ = nb.predict(X)
accuracy = np.mean(y_pred == y)
print(f"\n训练集准确率: {accuracy*100:.2f}%")

# 混淆矩阵
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

cm = confusion_matrix(y, y_pred)
print("\n混淆矩阵:")
print(f"          预测正常  预测垃圾")
print(f"  实际正常   {cm[0,0]}      {cm[0,1]}")
print(f"  实际垃圾   {cm[1,0]}      {cm[1,1]}")

precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
print(f"\n精确率: {precision*100:.2f}%")
print(f"召回率: {recall*100:.2f}%")

# ============================================================
# 5. 预测新邮件
# ============================================================
X_new = np.array([
    [1, 1, 1, 1],  # 含"免费""紧急""中奖"
    [0, 0, 0, 0],  # 都没有
    [1, 0, 1, 0],  # 含"免费""中奖"
    [0, 1, 0, 1]   # 含"紧急",长度长
])

preds, posteriors = nb.predict(X_new)
print("\n新邮件预测:")
for i, (x, p) in enumerate(zip(X_new, preds)):
    label = "🚫 垃圾邮件" if p == 1 else "✅ 正常邮件"
    # 转为概率
    post = np.exp(posteriors[i])
    post = post / np.sum(post)
    print(f"  邮件{i+1}: {label} (正常概率={post[0]:.3f}, 垃圾概率={post[1]:.3f})")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 混淆矩阵热力图
im = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['预测正常', '预测垃圾'])
axes[0].set_yticklabels(['实际正常', '实际垃圾'])

for i in range(2):
    for j in range(2):
        text = axes[0].text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=20)
axes[0].set_title('混淆矩阵', fontsize=14)
plt.colorbar(im, ax=axes[0])

# 图2: 各特征的条件概率对比
x_pos = np.arange(len(feature_names))
width = 0.35

# 找出垃圾邮件和正常邮件对应的索引
spam_idx = list(nb.classes).index(1)
ham_idx = list(nb.classes).index(0)

p_spam = nb.likelihoods[spam_idx, :, 1]
p_ham = nb.likelihoods[ham_idx, :, 1]

axes[1].bar(x_pos - width/2, p_spam, width, label='垃圾邮件', color='red', alpha=0.7)
axes[1].bar(x_pos + width/2, p_ham, width, label='正常邮件', color='green', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(feature_names, rotation=15)
axes[1].set_ylabel('P(特征=是|类别)', fontsize=12)
axes[1].set_title('各特征在不同类别下的概率', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 机器学习算法实战集

这个文件夹包含 8 个经典机器学习算法的完整实现，每个算法都有：
- 手动实现（不依赖 sklearn）
- 真实场景数据
- 可视化展示
- 模型评估
- 新数据预测

## 文件清单

```
MachineLearning/
├── linearReg.py            # 一元线性回归 (基础版)
├── multiLinearReg.py       # 多元线性回归 - 房价预测
├── logisticReg.py          # 逻辑回归 - 购买预测
├── knn.py                  # K近邻 - 水果分类
├── decisionTree.py         # 决策树 - 是否打球
├── kmeans.py               # K-Means - 客户分群
├── naiveBayes.py           # 朴素贝叶斯 - 垃圾邮件
├── svm.py                  # 支持向量机 - 大小分类
├── neuralNetwork.py        # 神经网络 - XOR
└── README.md
```

## 算法速查表

| 算法 | 文件 | 类别 | 场景 |
|------|------|------|------|
| 一元线性回归 | linearReg.py | 监督-回归 | 简单回归 |
| 多元线性回归 | multiLinearReg.py | 监督-回归 | 多特征回归 |
| 逻辑回归 | logisticReg.py | 监督-分类 | 二分类 |
| K近邻 | knn.py | 监督-分类 | 简单分类 |
| 决策树 | decisionTree.py | 监督-分类 | 可解释分类 |
| K-Means | kmeans.py | 无监督-聚类 | 客户分群 |
| 朴素贝叶斯 | naiveBayes.py | 监督-分类 | 文本分类 |
| SVM | svm.py | 监督-分类 | 高维分类 |
| 神经网络 | neuralNetwork.py | 监督-深度学习 | 复杂模式 |

## 运行方式

每个文件都可以独立运行：

```bash
python linearReg.py
python logisticReg.py
# ... 其他文件
```

确保已安装：

```bash
pip install numpy matplotlib
```

## 学习路线建议

```
第1周: linearReg.py + multiLinearReg.py  → 理解线性回归
第2周: logisticReg.py + knn.py           → 入门分类
第3周: decisionTree.py + naiveBayes.py   → 经典算法
第4周: kmeans.py                         → 入门无监督
第5周: svm.py                            → 进阶分类
第6周: neuralNetwork.py                  → 神经网络
```

## 每个文件包含

1. **数据准备** - 真实场景数据
2. **手动实现** - 不依赖库的算法
3. **训练过程** - 展示训练细节
4. **模型评估** - 准确率、误差等指标
5. **预测新数据** - 实际应用
6. **可视化** - 直观理解算法

"""单层感知机：用加权求和、阶跃函数和误差更新实现二分类。

运行：python perceptron.py
AND 和 OR 可以学会；XOR 无法被单个线性分类边界完全分开。
"""


class Perceptron:
    def __init__(self, input_size, learning_rate=1.0):
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        # 前向计算：加权求和，再用阶跃函数输出 0 或 1。
        z = sum(w * xi for w, xi in zip(self.weights, x, strict=True))
        z += self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, epochs=100):
        # X 是非空样本列表；y 是对应的 0/1 标签列表。
        self.converged_ = False
        self.epochs_ = 0
        for epoch in range(epochs):
            for x, target in zip(X, y, strict=True):
                error = target - self.predict(x)
                for j in range(len(self.weights)):
                    self.weights[j] += self.lr * error * x[j]
                self.bias += self.lr * error

            self.epochs_ = epoch + 1
            if all(self.predict(x) == target
                   for x, target in zip(X, y, strict=True)):
                self.converged_ = True
                break
        return self


def main():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    tasks = {
        "AND": [0, 0, 0, 1],
        "OR": [0, 1, 1, 1],
        "XOR": [0, 1, 1, 0],
    }

    for name, y in tasks.items():
        model = Perceptron(input_size=2).train(X, y)
        print(f"\n{name}: weights={model.weights}, bias={model.bias}")
        for x, target in zip(X, y, strict=True):
            print(f"  {x} -> {model.predict(x)} (target={target})")
        status = "all samples correct" if model.converged_ else "not converged"
        print(f"  {status}; epochs={model.epochs_}")


if __name__ == "__main__":
    main()

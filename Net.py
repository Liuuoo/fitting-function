import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid 的导数"""
    return x * (1 - x)


def readFile(filename):
    file_path = filename
    # 确保加载的是浮点型数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=np.float64)
    x_values = data[:, 0] / 100  # 归一化到 [0, 1]
    y_values = data[:, 1] / 2  # 归一化到 [0, 1]
    return x_values, y_values


def createEdge(now_node_size, next_node_size, filename):
    # Xavier 初始化
    edge = np.random.randn(now_node_size, next_node_size) * np.sqrt(2.0 / next_node_size)
    np.savetxt(filename, edge)


def readEdge(filename):
    edge_dir = Path(filename)
    if not edge_dir.exists():
        print("初始化权重...")
        edge_dir.mkdir(parents=True, exist_ok=True)
        createEdge(1, 100, edge_dir / "frontEdge.txt")
        createEdge(100, 100, edge_dir / "midEdge.txt")
        createEdge(100, 10, edge_dir / "backEdge.txt")
        createEdge(10, 1, edge_dir / "resultEdge.txt")
        return False


def calcLight(filename, x):
    readEdge(filename)

    # 加载并确保权重形状正确
    front_edge = np.loadtxt(f"{filename}/frontEdge.txt").reshape(1, 100)
    mid_edge = np.loadtxt(f"{filename}/midEdge.txt").reshape(100, 100)
    back_edge = np.loadtxt(f"{filename}/backEdge.txt").reshape(100, 10)
    result_edge = np.loadtxt(f"{filename}/resultEdge.txt").reshape(10, 1)

    # 前向传播
    input_data = np.array([[x]])  # 形状 (1, 1)

    node1 = sigmoid(np.dot(input_data, front_edge).T)  # (100, 1)
    node2 = sigmoid(np.dot(node1.T, mid_edge).T)  # (100, 1)
    node3 = sigmoid(np.dot(node2.T, back_edge).T)  # (10, 1)
    result = np.dot(node3.T, result_edge).T  # (1, 1)

    return node1, node2, node3, result


def backpropagate(left_act, right_act, delta, edge_weight, lr):
    """
    反向传播核心函数
    :param left_act:  前一层激活值 (m, 1)
    :param right_act: 当前层激活值 (n, 1)
    :param delta:     当前层梯度   (n, 1)
    :param edge_weight: 权重矩阵 (m, n)
    :param lr: 学习率
    :return: 新权重矩阵和前一层梯度
    """
    # 计算当前层梯度（包含激活函数导数）
    delta_current = delta * sigmoid_derivative(right_act)

    # 更新权重
    weight_update = lr * np.dot(left_act, delta_current.T)
    new_weights = edge_weight + weight_update

    # 计算前一层梯度
    delta_prev = np.dot(edge_weight, delta_current)

    return new_weights, delta_prev


# 训练流程
def train_one_sample(x, y_true, folder="测试", lr=0.1):
    # 前向传播
    node1, node2, node3, output = calcLight(folder, x)

    # 反向传播
    # 1. 输出层 -> 第三隐藏层
    delta_output = (output - y_true)  # 注意符号方向
    new_result_edge, delta3 = backpropagate(node3, output, delta_output,
                                            np.loadtxt(f"{folder}/resultEdge.txt").reshape(10, 1), lr)
    np.savetxt(f"{folder}/resultEdge.txt", new_result_edge)

    # 2. 第三隐藏层 -> 第二隐藏层
    new_back_edge, delta2 = backpropagate(node2, node3, delta3,
                                          np.loadtxt(f"{folder}/backEdge.txt").reshape(100, 10), lr)
    np.savetxt(f"{folder}/backEdge.txt", new_back_edge)

    # 3. 第二隐藏层 -> 第一隐藏层
    new_mid_edge, delta1 = backpropagate(node1, node2, delta2,
                                         np.loadtxt(f"{folder}/midEdge.txt").reshape(100, 100), lr)
    np.savetxt(f"{folder}/midEdge.txt", new_mid_edge)

    # 4. 第一隐藏层 -> 输入层
    input_data = np.array([[x]])
    new_front_edge, _ = backpropagate(input_data.T, node1, delta1,
                                      np.loadtxt(f"{folder}/frontEdge.txt").reshape(1, 100), lr)
    np.savetxt(f"{folder}/frontEdge.txt", new_front_edge)

# 画图并保存

def plot_and_save(x_values, y_values, predictions, folder="result", filename="plot.png"):

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, y_values, label="Original Function", color='blue', linestyle='--', linewidth=1)

    plt.plot(x_values, predictions, label="Predicted", color='red', linestyle='--', linewidth=1)

    # 添加标题和标签
    plt.title("Comparison between Original and Predicted Values")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()

    # 创建结果文件夹，如果不存在
    result_dir = Path(folder)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存图像
    plt.savefig(result_dir / filename)
    plt.close()

# 主程序
if __name__ == '__main__':
    x_values, y_values = readFile("sin_cos_training_data.csv")

    for i in range(len(x_values)):
        print(f"进行第{i}次训练...")
        # 每 400 次验证一次
        if i != 0 and i % 10 == 0:
            x_val, y_val = readFile("test.csv")

            predictions = []  # 存储预测结果
            for j in range(len(x_val)):
                _, _, _, val_output = calcLight("测试", x_val[j])
                predictions.append(val_output[0, 0])  # 获取预测结果

            # 绘制并保存图像
            plot_and_save(x_val, y_val, predictions, folder="result", filename=f"plot_epoch_{i}.png")

        # 训练单个样本
        train_one_sample(x_values[i], y_values[i], lr=0.5)

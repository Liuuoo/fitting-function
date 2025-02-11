import os
from matplotlib.ticker import FuncFormatter
import numpy as np
n_hidden = 16  # 隐藏层神经元数量
# 假设学习率eta
eta = 0.03

import matplotlib.pyplot as plt

# 设置中文字体，使用系统中支持的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是常见的黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def createData():
    # 数据生成
    np.random.seed(42)  # 固定随机种子，确保结果可重复

    # 生成训练数据
    n_train = 2000  # 训练样本数量
    x_train = np.linspace((-3/2)*np.pi, (1/2)*np.pi, n_train)
    y_train = np.sin(2 * x_train) + np.cos(5 * x_train)

    # 生成测试数据
    n_test = 100  # 测试样本数量
    x_test = np.linspace((-3/2)*np.pi, (1/2)*np.pi, n_test)
    y_test = np.sin(2 * x_test) + np.cos(5 * x_test)

    # 将数据标准化在[-1, 1]区间内
    x_scaled_train = (x_train - x_train.min()) / (x_train.max() - x_train.min()) * 2 - 1
    x_scaled_test = (x_test - x_test.min()) / (x_test.max() - x_test.min()) * 2 - 1
    y_test = y_test / 2
    y_train = y_train / 2

    # 将数据转为列向量
    x_train = x_scaled_train.reshape(-1, 1).astype(np.float64)
    y_train = y_train.reshape(-1, 1).astype(np.float64)
    x_test = x_scaled_test.reshape(-1, 1).astype(np.float64)
    y_test = y_test.reshape(-1, 1).astype(np.float64)

    # 保存数据到文件
    np.savez("training_data.npz",
             x_train=x_train,
             y_train=y_train,
             x_test=x_test,
             y_test=y_test)
    print("数据已保存到 training_data.npz 文件中。")

    # 初始化权重（He初始化）和偏置（初始化为零）
    # 第一层权重 (n_input, n_hidden)
    weights1 = np.random.randn(1, n_hidden) * np.sqrt(2. / (1 + n_hidden))
    bias1 = np.zeros((1, n_hidden))  # 偏置形状 (1, n_hidden)

    # 第二层权重 (n_hidden, n_hidden)
    weights2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(2. / (n_hidden + n_hidden))
    bias2 = np.zeros((1, n_hidden))  # 偏置形状 (1, n_hidden)

    # 第三层权重 (n_hidden, 1)
    weights3 = np.random.randn(n_hidden, 1) * np.sqrt(2. / (n_hidden + 1))
    bias3 = np.zeros((1, 1))  # 偏置形状 (1, 1)

    # 保存参数（包含偏置）
    np.savez("weights.npz",
             weights1=weights1, bias1=bias1,
             weights2=weights2, bias2=bias2,
             weights3=weights3, bias3=bias3)
    print("权重和偏置已保存到 weights.npz")

def getData():
    # 加载数据
    return np.load("training_data.npz")


def trainOneSimple(data, weights):

    loss = 0
    max_loss  =0
    # 加载权重和偏置
    weights1 = weights['weights1']
    bias1 = weights['bias1']  # 新增
    weights2 = weights['weights2']
    bias2 = weights['bias2']  # 新增
    weights3 = weights['weights3']
    bias3 = weights['bias3']  # 新增

    # 恢复数据
    x_train = data['x_train']
    y_train = data['y_train']

    for i in range(len(x_train)):
        # print(weights1)

        node0 = x_train[i].reshape(1, -1)  # 转为行向量 (1, input_size)
        # 前向传播（添加偏置）
        z1 = node0 @ weights1 + bias1  # 关键修改：+ bias1
        a1 = np.tanh(z1)
        z2 = a1 @ weights2 + bias2  # 关键修改：+ bias2
        a2 = np.tanh(z2)
        z3 = a2 @ weights3 + bias3  # 关键修改：+ bias3
        result = np.tanh(z3)

        # 计算损失
        loss += abs(result[0, 0] - y_train[i])
        max_loss = max(abs(result[0, 0] - y_train[i]), max_loss)

        # 反向传播（接前向传播代码）
        delta3 = (result - y_train[i]) * (1 - result ** 2)
        delta2 = delta3 @ weights3.T * (1 - a2 ** 2)
        delta1 = delta2 @ weights2.T * (1 - a1 ** 2)

        # 更新权重和偏置
        weights3 -= eta * a2.T @ delta3
        bias3 -= eta * np.sum(delta3, axis=0)  # 新增：偏置更新
        weights2 -= eta * a1.T @ delta2
        bias2 -= eta * np.sum(delta2, axis=0)  # 新增：偏置更新
        weights1 -= eta * node0.T @ delta1
        bias1 -= eta * np.sum(delta1, axis=0)  # 新增：偏置更新

    print("平均误差：",loss / len(x_train))
    print("最大误差", max_loss)
    # 更新权重字典
    weights['weights1'] = weights1
    weights['weights2'] = weights2
    weights['weights3'] = weights3
    weights['bias3'] = bias3
    weights['bias2'] = bias2
    weights['bias1'] = bias1

    return weights

def predict(x, weights):
    # 前向传播计算预测值
    weights1 = weights['weights1']
    weights2 = weights['weights2']
    weights3 = weights['weights3']
    bias3 = weights['bias3']
    bias2 = weights['bias2']
    bias1 = weights['bias1']

    y_pred = []
    for i in range(len(x)):
        node0 = x[i].reshape(1, -1)  # 转为行向量 (1, input_size)
        # 前向传播（添加偏置）
        z1 = node0 @ weights1 + bias1  # 关键修改：+ bias1
        a1 = np.tanh(z1)
        z2 = a1 @ weights2 + bias2  # 关键修改：+ bias2
        a2 = np.tanh(z2)
        z3 = a2 @ weights3 + bias3  # 关键修改：+ bias3
        y_pred.append(np.tanh(z3))
    return y_pred



def save_plot(weights, epoch, data):
    """绘制并保存拟合效果图"""
    os.makedirs('result', exist_ok=True)  # 确保目录存在

    x_test = data['x_test']
    y_test = data['y_test']

    # 生成预测结果
    y_pred_scaled = predict(x_test, weights)
    y_pred_scaled = np.array([y_pred_scaled])
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    # 计算原始数据的最大值和最小值
    x_train_original = np.linspace((-3/2)*np.pi, (1/2)*np.pi, 2000)
    x_test_original_max = x_train_original.max()
    x_test_original_min = x_train_original.min()

    # 数据范围逆标准化
    x_test_original = (x_test + 1) * (x_test_original_max - x_test_original_min) / 2 + x_test_original_min
    y_test_original = y_test * 2
    y_pred_original = y_pred_scaled * 2

    # 创建新图表
    plt.figure(figsize=(10, 6))
    plt.plot(x_test_original, y_test_original,
             label='真实值', linestyle='--', color='b')
    plt.plot(x_test_original, y_pred_original,
             label='预测值', linestyle='-', color='r')
    plt.title(f"模型拟合效果对比 (Epoch {epoch})")
    plt.xlabel("x (以π为单位)")
    plt.ylabel("y (原始范围)")

    # 设置 x 轴刻度和标签
    x_ticks = [-1.5 * np.pi, -np.pi, -0.5 * np.pi, 0, 0.5 * np.pi]
    x_tick_labels = ['-1.5π', '-π', '-0.5π', '0', '0.5π']

    # 设置刻度和标签
    plt.xticks(x_ticks, x_tick_labels)

    plt.legend()
    plt.grid(True)

    # 保存文件
    filename = f"result/epoch_{epoch:06d}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # 关闭图表释放内存




# 创建数据
createData()  # 确保数据文件存在

# 初始化权重
weights_file = np.load("weights.npz")
weights_dict = {
    'weights1': weights_file['weights1'],
    'weights2': weights_file['weights2'],
    'weights3': weights_file['weights3'],
    'bias3': weights_file['bias3'],
    'bias2': weights_file['bias2'],
    'bias1': weights_file['bias1'],
}
# 获取数据
data = getData()
# print(data['x_train'])
# print(weights_file['weights1'])
# 训练循环
for epoch in range(50000):
    print(f"\n开始训练epoch {epoch + 1}")

    updated_weights = trainOneSimple(data, weights_dict)

    # 保存权重
    np.savez("weights.npz",
             weights1=updated_weights['weights1'],
             weights2=updated_weights['weights2'],
             weights3=updated_weights['weights3'],
             bias1=updated_weights['bias1'],
             bias2=updated_weights['bias2'],
             bias3=updated_weights['bias3'],)

    # 每1000次保存一张图片
    if epoch % 1000 == 0:
        save_plot(updated_weights, epoch + 1, data)
        print(f"已保存第 {epoch + 1} 次迭代结果图")

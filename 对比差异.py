import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取 CSV 文件
df = pd.read_csv("sin_cos_training_data.csv")  # 请替换为你的文件路径
x_csv = df["x"]
y_csv = df["y"]

# 重新计算 y = sin(2x) + cos(5x)
x_calc = np.linspace(0, 2 * np.pi, 1000)
y_calc = np.sin(2 * x_calc) + np.cos(5 * x_calc)

# 绘制曲线（使用不同虚线样式）
plt.figure(figsize=(8, 4))
plt.plot(x_csv, y_csv, label="CSV 数据", color="red", linestyle="dashed")   # 红色短虚线 "--"
plt.plot(x_calc, y_calc, label="直接计算", color="blue", linestyle="dotted")  # 蓝色点状线 ":"

# 添加标签和标题
plt.xlabel("x")
plt.ylabel("y")
plt.title("CSV 数据 vs. 直接计算（不同虚线样式）")
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig("comparison_plot.png", dpi=300)
plt.show()

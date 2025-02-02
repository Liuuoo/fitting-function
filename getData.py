import numpy as np
import pandas as pd

# 设定周期
T = 2 * np.pi

# 生成 3000 个 x 值，范围从 0 到 T
x_values = np.linspace(-10*T, 10*T, 3000)

# 计算对应的 y 值，并保留 4 位小数
y_values = np.round(np.sin(2 * x_values) + np.cos(5 * x_values), 4)

# 创建 DataFrame
df = pd.DataFrame({"x": x_values, "y": y_values})

# 保存为 CSV 文件
file_path = "test.csv"  # 本地保存路径
df.to_csv(file_path, index=False, encoding="utf-8")

print(f"CSV 文件已生成: {file_path}")

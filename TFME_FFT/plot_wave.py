import numpy as np
import matplotlib.pyplot as plt

TXT = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01_Timeseries\C016ST01_#Period=01_Timeseries.txt"

# === 修改为你实际生成的 TXT 文件路径 ===
txt_file = TXT

# 跳过前两行表头，读取数据
data = np.loadtxt(txt_file, skiprows=2)

# 在接收机(RX)文件中，第 0 列是通道1(Ex)，第 1 列是通道2(Ey)
# 发射机(TX)，第 1 列是通道2(电流)
channel_to_plot = 1

# 提取通道数据
waveform = data[:, channel_to_plot]
plot_points = 1000

# 显式生成横坐标，让它从 1 开始，到 plot_points 结束
x_axis = np.arange(1, plot_points + 1)

plt.figure(figsize=(15.8, 4))

# 将 x_axis 作为第一个参数传入
plt.plot(x_axis, waveform[:plot_points], marker='.', linestyle='-', color='b')

plt.title(f"Waveform - Period 1 - Channel {channel_to_plot + 1}")
plt.xlabel("Sample Points")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
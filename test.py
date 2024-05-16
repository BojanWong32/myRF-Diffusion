# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
#
# mat_data = sio.loadmat('dataset/wifi/output/batch-1-0.mat')
# complex_matrix = mat_data['pred']
# amplitude_matrix = np.abs(complex_matrix)
#
# amplitude_matrix = np.transpose(amplitude_matrix)
#
# for i in range(amplitude_matrix.shape[0]):
#     if i < 30:
#         color = 'red'  # 设置前30条线的颜色为红色
#     elif i < 60:
#         color = 'green'  # 设置中间30条线的颜色为绿色
#     else:
#         color = 'blue'  # 设置后40条线的颜色为蓝色
#     plt.plot(amplitude_matrix[i], color=color)
#
# plt.title('CSI Waveforms')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 准备数据
# x = [0, 0.25, 0.5, 0.75, 1.00]
# y = [0.2954, 0.4465, 0.5612, 0.5477, 0.5292]
#
# # 绘制折线图
# plt.plot(x, y)
#
# # 添加标题和坐标轴标签
# plt.title('折线图示例')
# plt.xlabel('数据增强量')
# plt.ylabel('准确度')
#
# # 显示图形
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['SimHei']

# 准备数据
x = [0, 0.25, 0.5, 0.75, 1.00]
y = [0.2954, 0.4465, 0.5612, 0.5477, 0.5292]

# x = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
# y = [0.2954, 0.4465, 0.5612, 0.5477, 0.5292]

# y = [0.29, 0.27, 0.35,0.34,0.32, 0.44]
# 绘制折线图
plt.plot(x, y)

# 添加标题和坐标轴标签
plt.title('折线图示例')
plt.xlabel('数据增强量')
plt.ylabel('准确度')

# 自定义横轴刻度标签
x_labels = ['+0%', '+25%', '+50%', '+75%', '+100%']
# x_labels = ['+0%', '+5%', '+10%', '+15%', '+20%', '+25%']
plt.xticks(x, x_labels)

# 在纵轴上画虚线
plt.axhline(y=0.3, color='gray', linestyle='--')
plt.axhline(y=0.35, color='gray', linestyle='--')
plt.axhline(y=0.40, color='gray', linestyle='--')
plt.axhline(y=0.45, color='gray', linestyle='--')
plt.axhline(y=0.50, color='gray', linestyle='--')
plt.axhline(y=0.55, color='gray', linestyle='--')
# plt.axhline(y=0.425, color='gray', linestyle='--')

# 在横轴上画虚线
plt.axvline(x=0.0, color='gray', linestyle='--')
plt.axvline(x=0.25, color='gray', linestyle='--')
plt.axvline(x=0.50, color='gray', linestyle='--')
plt.axvline(x=0.75, color='gray', linestyle='--')
plt.axvline(x=1, color='gray', linestyle='--')
# plt.axvline(x=0.25, color='gray', linestyle='--')

plt.scatter(x, y, marker='x', color='red', s=50)

# 显示图形
plt.show()

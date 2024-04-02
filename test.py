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
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
# 准备数据
x = [0, 0.05, 0.1, 0.15, 0.20, 0.25]
y = [0.9623, 0.9339, 0.9257, 0.8962, 0.8991, 0.89573]

# 绘制折线图
plt.plot(x, y)

# 添加标题和坐标轴标签
plt.title('折线图示例')
plt.xlabel('合成数据与真实数据比值')
plt.ylabel('无限感知系统准确度')

# 显示图形
plt.show()
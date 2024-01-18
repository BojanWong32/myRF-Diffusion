import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

mat_data = sio.loadmat('dataset/wifi/output/batch-1-0.mat')
complex_matrix = mat_data['pred']
amplitude_matrix = np.abs(complex_matrix)

amplitude_matrix = np.transpose(amplitude_matrix)

for i in range(amplitude_matrix.shape[0]):
    if i < 30:
        color = 'red'  # 设置前30条线的颜色为红色
    elif i < 60:
        color = 'green'  # 设置中间30条线的颜色为绿色
    else:
        color = 'blue'  # 设置后40条线的颜色为蓝色
    plt.plot(amplitude_matrix[i], color=color)

plt.title('CSI Waveforms')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
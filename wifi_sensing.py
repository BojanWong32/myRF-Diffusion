import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from scipy.io import loadmat

# 步骤1：准备数据
data_folder = '文件夹路径'  # 替换为实际的文件夹路径
num_classes = 6  # 替换为实际的类别数

# 加载数据和标签
X = []
Y = []
file_list = os.listdir(data_folder)
for file_name in file_list:
    file_path = os.path.join(data_folder, file_name)
    data = loadmat(file_path)['data']  # 假设.mat文件中的数据存储在"data"字段中
    label = file_name[:-4]  # 假设文件名是标签的一部分，去掉后缀".mat"
    X.append(data)
    Y.append(label)

# 将数据和标签转换为NumPy数组
X = np.array(X)
Y = np.array(Y)

# 将标签转换为one-hot编码
Y = to_categorical(Y, num_classes)

# 步骤2：划分训练集和测试集
train_ratio = 0.8  # 训练集比例
test_ratio = 0.2  # 测试集比例

num_data = X.shape[0]
indices = np.random.permutation(num_data)
train_indices = indices[:round(train_ratio * num_data)]
test_indices = indices[round(train_ratio * num_data):]

X_train = X[train_indices]
Y_train = Y[train_indices]
X_test = X[test_indices]
Y_test = Y[test_indices]

# 步骤3：构建神经网络模型
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(1, 512, 90)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# 步骤4：编译和训练模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test))

# 步骤5：测试准确度
_, train_accuracy = model.evaluate(X_train, Y_train, verbose=0)
_, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

print('训练集准确度:', train_accuracy)
print('测试集准确度:', test_accuracy)
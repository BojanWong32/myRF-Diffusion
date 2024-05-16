import os
import random
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from keras import layers
import re

data_folder = 'dataset/wifi/train_dataset'  # 750*4
new_data_folder1 = 'dataset/wifi/output_4_5_2'  # 750
new_data_folder2 = 'dataset/wifi/output_4_5_3'  # 750
new_data_folder3 = 'dataset/wifi/output_4_5_4'  # 750
new_data_folder4 = 'dataset/wifi/output_4_5_5'  # 750

save_path = 'sensing_model/model_4'

index = 5


data = []
labels = []

for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

for file_name in os.listdir(new_data_folder1):
    file_path = os.path.join(new_data_folder1, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1 or repetition > index:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

for file_name in os.listdir(new_data_folder2):
    file_path = os.path.join(new_data_folder2, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1 or repetition > index:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

for file_name in os.listdir(new_data_folder3):
    file_path = os.path.join(new_data_folder3, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1 or repetition > index:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)

for file_name in os.listdir(new_data_folder4):
    file_path = os.path.join(new_data_folder4, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        if receiver != 1 or repetition > index:  # 只接受r1
            continue

        mat_data = loadmat(file_path)

        data.append(mat_data['pred'].reshape(512, 90))
        labels.append(gesture-1)


data = np.array(data)

real_data = np.real(data).astype(np.float32)
imag_data = np.imag(data).astype(np.float32)
data = np.stack((real_data, imag_data), axis=-1)

# print(data.shape)
labels = np.array(labels)

print(data.shape)

data_size = len(data)
indices = np.arange(data_size)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

train_ratio = 0.8  # 训练集比例
train_size = int(data_size * train_ratio)

train_data = data[:train_size]
train_labels = labels[:train_size]
test_data = data[train_size:]
test_labels = labels[train_size:]

train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)


num_classes = 6
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

input_shape = (512, 90, 2)
# print(input_shape)
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 10
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save(save_path)

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)


# 0.2954925000667572
# 0.2750397324562073
# 0.34901365637779236
# 0.3294629752635956
# 0.31476321816444397
# 0.446524053812027  %25

# 虚数分离后
# 0.3388981521129608
# 0.4131016135215759
# 0.5400890707969666

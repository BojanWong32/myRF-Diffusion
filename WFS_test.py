import os
import random
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from keras import layers
import re

load_path = 'sensing_model/model_2'
test_dataset = 'dataset/wifi/test_dataset2'

model = tf.keras.models.load_model(load_path)
test_data = []
test_labels = []

for file_name in os.listdir(test_dataset):
    file_path = os.path.join(test_dataset, file_name)
    if file_path.endswith('.mat'):
        gesture = int(file_name.split('-')[1])
        location = int(file_name.split('-')[2])
        orientation = int(file_name.split('-')[3])
        repetition = int(file_name.split('-')[4])
        # print(file_name)
        receiver = int(file_name.split('r')[2].split('.')[0])
        # print(receiver)

        mat_data = loadmat(file_path)

        test_data.append(mat_data['pred'].reshape(512, 90))
        test_labels.append(gesture - 1)

test_data = np.array(test_data)

real_data = np.real(test_data).astype(np.float32)
imag_data = np.imag(test_data).astype(np.float32)
test_data = np.stack((real_data, imag_data), axis=-1)

test_labels = np.array(test_labels)

test_data = np.expand_dims(test_data, axis=-1)

num_classes = 6
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

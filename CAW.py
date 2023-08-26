import tensorflow as tf
import shutil
import numpy as np
import pandas as pd
import cv2
import os
import time
import matplotlib.pyplot as plt
import glob

import efficientnet.tfkeras as enet
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
tf.compat.v1.enable_eager_execution()

''' Seed for reproducible results '''
# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value

''' Read labels '''
all_files = glob.glob("resized_dataset/*.png")
# all_files = np.random.choice(all_files, int(len(all_files) * 0.6), replace=False)
# all_files = all_files[int(len(all_files) * 0.2):int(len(all_files) * 0.5)]
all_files = all_files[int(len(all_files) * 0.7):]
# labels = np.zeros(len(all_files), dtype=np.float)
labels = []
for i in range(len(all_files)):
    file = all_files[i].split('\\')[1]
#     print(file)
    file = file.split('.')[0]
#     labels[i] = file.split('_')[1]
    labels.append(file.split('_')[1])
print(labels)
labels = np.array(labels)
print(labels.shape)
labels[labels == 'Normal'] = 'Benign'
print(labels)

labels[labels == 'Benign'] = int(0.0)
labels[labels == 'Cancer'] = int(1.0)
print(labels)

''' Read images '''
dataset = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in all_files], dtype=np.float)
print(dataset.shape)
plt.imshow(dataset[0], cmap='gray')
plt.show()

print('Normalization')
dataset = dataset.astype('float32')
print(np.max(dataset))
print(np.min(dataset))
dataset /= np.float(255)
# dataset -= 0.5
# dataset *= 2
print(np.max(dataset))
print(np.min(dataset))

dataset = np.stack((dataset,) * 3, axis=-1)
x_test_1 = dataset
y_test1 = labels


''' Categorical labels '''
from tensorflow.keras.utils import to_categorical

# y_train1 = to_categorical(y_train1, num_classes=len(np.unique(labels)))
# y_test1 = to_categorical(y_test1, num_classes=len(np.unique(labels)))

def buil_model_function(model_name):
    if model_name == 'enet':
        model = enet.EfficientNetB3(include_top=False, pooling='avg', input_shape=(400, 250, 3))
        x = model.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # output layer
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        model_final.load_weights('enetb3_ddsm_bahman_1.h5')
        print(model_name)
        model_final.summary()
        return model_final

    elif model_name == 'Xception':
        model = tf.keras.applications.Xception(include_top=False,
                                               pooling='avg',
                                               input_shape=(400, 250, 3),
                                               weights='imagenet')
        x = model.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # output layer
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        model_final.load_weights('Xception_ddsm_bahman_1.h5')
        print(model_name)
        model_final.summary()
        return model_final

    elif model_name == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2(include_top=False,
                                                  pooling='avg',
                                                  input_shape=(400, 250, 3),
                                                  weights='imagenet')
        x = model.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # output layer
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        model_final.load_weights('MobileNetV2_ddsm_bahman_1.h5')
        print(model_name)
        model_final.summary()
        return model_final

    elif model_name == 'InceptionV3':
        model = tf.keras.applications.InceptionV3(include_top=False,
                                                  pooling='avg',
                                                  input_shape=(400, 250, 3),
                                                  weights='imagenet')
        x = model.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # output layer
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        model_final.load_weights('InceptionV3_ddsm_bahman_1.h5')
        print(model_name)
        model_final.summary()
        return model_final

    elif model_name == 'ResNet50':
        model = tf.keras.applications.ResNet50(include_top=False,
                                               pooling='avg',
                                               input_shape=(400, 250, 3),
                                               weights='imagenet')
        x = model.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # output layer
        predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
        model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

        model_final.load_weights('ResNet50_ddsm_bahman_1.h5')
        print(model_name)
        model_final.summary()
        return model_final
    else:
        return 'Invalid model name'


model_final1 = buil_model_function('enet')
model_final2 = buil_model_function('Xception')
model_final3 = buil_model_function('MobileNetV2')
model_final4 = buil_model_function('InceptionV3')
model_final5 = buil_model_function('ResNet50')
# model_final.summary()

models = [model_final1, model_final2, model_final3, model_final4, model_final5]

preds = [model.predict(x_test_1) for model in models]

preds = np.array(preds)

print(preds.shape)
y_test1 = np.array(y_test1).astype('float32')

from sklearn.metrics import accuracy_score
# prediction1 = model_final1.predict_classes(x_test_1)
# prediction2 = model_final2.predict_classes(x_test_1)
# prediction3 = model_final3.predict_classes(x_test_1)
# prediction4 = model_final4.predict_classes(x_test_1)
# prediction5 = model_final5.predict_classes(x_test_1)
all_predictions = np.argmax(preds, axis=2)
accuracy1 = accuracy_score(y_test1, all_predictions[0])
accuracy2 = accuracy_score(y_test1, all_predictions[1])
accuracy3 = accuracy_score(y_test1, all_predictions[2])
accuracy4 = accuracy_score(y_test1, all_predictions[3])
accuracy5 = accuracy_score(y_test1, all_predictions[4])
print(preds.shape)
# print(np.argmax(preds, axis=2).shape)

from sklearn.metrics import accuracy_score

def weighted_classes(preds, ideal_weights):
    ideal_weights_preds = np.tensordot(preds, ideal_weights, axes= ((0), (0)))
    ideal_weights_ensemble_predictions = np.argmax(ideal_weights_preds, axis = 1)
    return ideal_weights_ensemble_predictions
ideal_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
# ideal_weights_preds = np.tensordot(preds, ideal_weights, axes= ((0), (0)))
# ideal_weights_ensemble_predictions = np.argmax(ideal_weights_preds, axis = 1)
ideal_weights_ensemble_predictions = weighted_classes(preds, ideal_weights)
print(ideal_weights_ensemble_predictions)
print(y_test1)
ideal_weighted_accuracy = accuracy_score(y_test1, ideal_weights_ensemble_predictions)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for model4 = ', accuracy4)
print('Accuracy Score for model5 = ', accuracy5)
print('Accuracy Score for average ensemble = ', ideal_weighted_accuracy)

from sklearn.metrics import roc_auc_score


def results(y_test1, y_pred):
    acc = accuracy_score(y_test1, y_pred)
    cm = confusion_matrix(y_test1, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test1, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    f2_score = 5 * (precision * recall) / (4 * precision + recall)
    return acc, cm, precision, recall, f1_score, f2_score


acc_list = np.zeros(len(all_predictions))
precision_list = np.zeros(len(all_predictions))
recall_list = np.zeros(len(all_predictions))
f1_score_list = np.zeros(len(all_predictions))
f2_score_list = np.zeros(len(all_predictions))
for i in range(len(all_predictions)):
    acc, cm, precision, recall, f1_score, f2_score = results(y_test1, all_predictions[i])
    print(acc, cm, precision, recall, f1_score, f2_score)
    acc_list[i] = acc
    precision_list[i] = precision
    recall_list[i] = recall
    f1_score_list[i] = f1_score
    f2_score_list[i] = f2_score

print('\nMy model:')
# for step in range(0.5, 3, 0.1):
#     print(step)
acc_plot = []
precision_plot = []
recall_plot = []
f1_score_plot = []
f2_score_plot = []
steps = np.arange(0.5, 30, 0.1)
for i in steps:
    ideal_weights = np.power(precision_list, i) / np.sum(np.power(precision_list, i))
    # print(ideal_weights)
    # ideal_weights = precision_list / np.sum(precision_list)
    # print(acc_list)
    # print(np.sum(acc_list))
    # print(np.sum(f1_score_list))
    # print(np.sum(np.power(acc_list, 2)))
    # print(np.sum(np.power(f1_score_list, 2)))
    # print(f1_score_list)
    # ideal_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    # ideal_weights = ideal_weights / np.sum(ideal_weights)
    # print(ideal_weights)

    ideal_weights_ensemble_predictions = weighted_classes(preds, ideal_weights)
    acc, cm, precision, recall, f1_score, f2_score = results(y_test1, ideal_weights_ensemble_predictions)
    acc_plot.append(acc)
    precision_plot.append(precision)
    recall_plot.append(recall)
    f1_score_plot.append(f1_score)
    f2_score_plot.append(f2_score)
    print(i)
    print(acc, cm, precision, recall, f1_score, f2_score)
    print('AUC: ', roc_auc_score(y_test1, ideal_weights_ensemble_predictions))
plt.plot(steps, acc_plot, label='acc')

plt.plot(steps, precision_plot, label='precision')
plt.plot(steps, recall_plot, label='recall')
plt.plot(steps, f1_score_plot, label='f1_score')
plt.plot(steps, f2_score_plot, label='f2_score')

plt.grid(True)
plt.legend()
plt.show()

# ideal_weights_max = ideal_weights
ideal_weights_mid = ideal_weights
print(ideal_weights_mid)


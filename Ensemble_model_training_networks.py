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
all_files = glob.glob("data/*.jpg")
# all_files = np.random.choice(all_files, int(len(all_files) * 0.6), replace=False)
all_files = all_files[:int(len(all_files) * 0.9)]
labels = np.zeros(len(all_files), dtype=np.float)
for i in range(len(all_files)):
    file = all_files[i].split('\\')[1]
    file = file.split('.')[0]
    labels[i] = int(file.split('_')[1])
print(labels)
print(labels.shape)

''' Read images '''
dataset = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in all_files], dtype=np.float)
print(dataset.shape)

a = labels[500:8000]
print(np.unique(a, return_counts=True))

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

''' Train test split '''
x_train_1, x_test_1, y_train1, y_test1 = train_test_split(dataset, labels, test_size=0.15, random_state=0)  # , shuffle=True)
a = y_train1[1000:4000]
print(np.unique(a, return_counts=True))


''' Categorical labels '''
from tensorflow.keras.utils import to_categorical

y_train1 = to_categorical(y_train1, num_classes=len(np.unique(labels)))
y_test1 = to_categorical(y_test1, num_classes=len(np.unique(labels)))

''' Build model '''
# loading B0 pre-trained on ImageNet without final aka fiature extractor
# model = enet.EfficientNetB3(include_top=False, pooling='avg', input_shape=(120, 120, 3), weights='imagenet')
# model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(120, 120, 3))
# model = tf.keras.applications.Xception(include_top=False,
#                                           pooling='avg',
#                                           input_shape=(120, 120, 3),
#                                           weights='imagenet')
model = tf.keras.applications.ResNet50(include_top=False,
                                          pooling='avg',
                                          input_shape=(120, 120, 3),
                                          weights='imagenet')
# model = tf.keras.applications.InceptionV3(include_top=False,
#                                           pooling='avg',
#                                           input_shape=(120, 120, 3),
#                                           weights='imagenet')
# model = tf.keras.applications.MobileNetV2(include_top=False,
#                                           pooling='avg',
#                                           input_shape=(120, 120, 3),
#                                           weights='imagenet')

x = model.output

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)

# output layer
predictions = tf.keras.layers.Dense(2, activation="softmax")(x)

model_final = tf.keras.models.Model(inputs=model.input, outputs=predictions)

model_final.summary()

''' Plot online results (Live results)'''
from IPython.display import clear_output


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acces = []
        self.val_acces = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acces.append(logs.get('acc'))
        self.val_acces.append(logs.get('val_acc'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        print(self.x)
        print(self.losses)
        print(self.val_losses)
        plt.ylim((0, 4))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        ''' Validation Accuracy plot '''
        # plt.plot(self.x, self.acces, label="acc")
        # plt.plot(self.x, self.val_acces, label="val_acc")
        # plt.legend()
        # plt.grid(True)
        # plt.show()


plot_losses = PlotLosses()


''' Train model '''
start_time = time.time()

# model compilation
# Memory check
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# sess = tf.Session(config=config)
# set_session(sess)

# sgd = tf.keras.optimizers.SGD(lr=0.0007, decay=1e-6, momentum=0.8, nesterov=False)
# optm = AdaBound(lr=1e-03,
#                 final_lr=0.1,
#                 gamma=1e-03,
#                 weight_decay=0.,
#                 amsbound=False)
# optm = AdaBound(lr=8*1e-04,
#                 final_lr=0.1,
#                 gamma=1e-03)
# from adabound_me import AdaBoundOptimizer
# optm = AdaBoundOptimizer(amsbound=True,
#                          learning_rate=0.001,
#                          final_lr=0.1,
#                          beta1=0.9,
#                          beta2=0.999,
#                          gamma=1e-3,
#                          epsilon=1e-8,
#                          use_locking=False)
optm = tf.keras.optimizers.Adam(learning_rate=10e-05)
# optm = tf.keras.optimizers.Adam()
# optm = Adadelta()
# optm = adagrad()
# optm = SGD()
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
model_final.compile(loss='binary_crossentropy',
                    optimizer=optm,
                    metrics=['accuracy', tf.keras.metrics.AUC(), 'binary_accuracy'])

mcp_save = tf.keras.callbacks.ModelCheckpoint('Xception_fusion.h5', save_best_only=True, monitor='val_loss')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, )

# print("Training....")
history = model_final.fit(x_train_1, y_train1,
                          batch_size=28,
                          epochs=30,
                          validation_split=0.1,
                          callbacks=[plot_losses, mcp_save],#, reduce_lr],
                          # shuffle=True,
                          verbose=1)
# history = model.fit(x_train_1, y_train1,
#                     validation_data=(x_test_1, y_test1),
#                     callbacks=[plot_losses],
#                     epochs=10, batch_size=4, shuffle=True)


''' Results '''
# eval = model.evaluate(x=x_test_1, y=y_test1)

print(history.history)

# print(eval)

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['Train', 'Test'], loc='lower right')
plt.ylim((0, 1))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['Train', 'Test'], loc='upper right')
plt.ylim((0, 4))
plt.show()

evaluation = model_final.evaluate(x=x_test_1, y=y_test1)
print("test loss, test acc:", evaluation)
y_pred = model_final.predict(x_test_1)
y_pred = np.argmax(y_pred, axis=1)
y_test11 = np.argmax(y_test1, axis=1)
cm = confusion_matrix(y_test11, y_pred)
print('Confusion Matrix')
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test11, y_pred).ravel()

precision = tp / (tp + fp)
print("Precision: {}\n".format(precision))
recall = tp / (tp + fn)
print("Recall: {}\n".format(recall))

print('\nTN: {} \n FP: {} \n FN: {} \n TP: {} \n'.format(tn, fp, fn, tp))

print('Accuracy')
print(history.history['val_binary_accuracy'])
print(np.max(history.history['val_binary_accuracy']))

# print('AUC')
# print(history.history['val_auc'])
# print(np.max(history.history['val_auc']))

f1_score = 2 * (precision * recall) / (precision + recall)
print('f1_score: {}'.format(f1_score))


''' Best model weight results '''
print('\n\n ****** Best model weight results: ****** \n\n\n')
model_final.load_weights('Xception_fusion.h5')

'''Compile model '''
# from adabound_me import AdaBoundOptimizer
# optm = AdaBoundOptimizer(amsbound=False,
#                          learning_rate=0.001,
#                          final_lr=0.1,
#                          beta1=0.9,
#                          beta2=0.999,
#                          gamma=1e-3,
#                          epsilon=1e-8,
#                          use_locking=False)
# optm = tf.keras.optimizers.Adam()
model_final.compile(loss='binary_crossentropy',
                    optimizer=optm,
                    metrics=['accuracy', tf.keras.metrics.AUC()])

evaluation = model_final.evaluate(x=x_test_1, y=y_test1)
print("test loss, test acc:", evaluation)
y_pred = model_final.predict(x_test_1)
y_pred = np.argmax(y_pred, axis=1)
y_test12 = np.argmax(y_test1, axis=1)
cm = confusion_matrix(y_test12, y_pred)
print('Confusion Matrix')
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test12, y_pred).ravel()

precision = tp / (tp + fp)
print("Precision: {}\n".format(precision))
recall = tp / (tp + fn)
print("Recall: {}\n".format(recall))

print('\nTN: {} \n FP: {} \n FN: {} \n TP: {} \n'.format(tn, fp, fn, tp))


print('Accuracy')
print(history.history['val_binary_accuracy'])
print(np.max(history.history['val_binary_accuracy']))

# print('AUC')
# # print(history.history['val_auc'])
# print(np.max(history.history['val_auc']))

f1_score = 2 * (precision * recall) / (precision + recall)
print('f1_score: {}'.format(f1_score))

print("--- %s seconds ---" % (time.time() - start_time))


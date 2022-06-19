import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
import Clases
import cv2
import os
import palettable
import numpy as np
mnist = tf.keras.datasets.mnist

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_l=y_test

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0

model= tf.keras.models.load_model('saved_model/my_model_LR.h5')
P=list()
success=np.zeros(10)


directory='C:/Users/apari/Desktop/Generated Data/'
xx=[np.linalg.norm(i) for i in x_test]
I=np.zeros(450)
k=0
for image_path in os.listdir(directory):
        input_path = os.path.join(directory, image_path)
        x = cv2.imread(input_path, 0)
        x = x / 255
        v=np.linalg.norm(x)
        xx=[np.abs(v-i) for i in xx]
        I[k]=xx.index(min(xx))
        k+=1


err=np.zeros(9)
success=np.zeros(9)

for i in range(1,10):
        k = 0
        l = 0
        for image_path in os.listdir(directory):
                input_path = os.path.join(directory, image_path)
                x = cv2.imread(input_path,0)
                x=x/255
                pred=np.argmax(model.predict(x.reshape(1, 28*28)))
                p = [k for k, letter in enumerate(image_path) if letter == '_']
                label=image_path[:p[0]]

                if label == str(i):
                     success[i-1] += (str(pred) != label)
                     err[i-1]+=np.linalg.norm(x_test[int(I[k])]-x)
                     l+=1
        k+=1
err=err/l
success=success/l

fig, ax1 = plt.subplots()
ax1.set_prop_cycle('color', palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
ax2 = ax1.twinx()
plt.xticks(range(1,10))
ax1.plot(range(1,10),success,'C1-')
ax2.plot(range(1,10),err,'C0-')
ax1.set_xlabel('Hyper-Parameter')
ax1.set_ylabel('Success Rate', color='C1')
ax2.set_ylabel('AbsoluteError', color='C0')
ax1.legend(['Success'], loc='upper right')
ax2.legend(['Error'], loc='lower left')


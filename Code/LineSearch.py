import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
import Clases
import numpy as np
import time
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_l=y_test

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0

model= tf.keras.models.load_model('saved_model/my_model_LR.h5')

#print(np.argmax(model.predict(x_test[1].reshape(1,28,28,1))))

for i in [0]:
    l=True
    yt = y_l[y_l == i]
    xt = x_test[y_l == i]

    R=np.zeros((10,3))
    for j in np.linspace(0.01,0.1,10):
        N = 1
        r=0

        for p in range(0,5):
            k = np.random.randint(0, len(xt) - 1)

            while l:
                k = np.random.randint(0, len(xt) - 1)
                if yt[k] == np.argmax(model.predict(xt[k].reshape(1, 28*28))):
                    l = False

            success = 0

            start = time.time()
            sol = Clases.WhiteBox(xt[k].reshape(1, 28*28), ["Untargeted", 'saved_model/my_model_LR.h5', "LM", "NN", j]).xp
            end = time.time()

            if np.argmax(model.predict(sol.x.reshape(1, 28*28))) == yt[k]:
                success = 1
            cv2.imwrite((str(yt[k])+'_'+str(j)+'_'+ str(p) + '.png'),255*sol.x.reshape(28, 28))

            R[r, 0] = ((end - start)+R[r,0])
            R[r, 1] = (success+R[r,1])
            R[r,2]=R[r,2]+np.linalg.norm(sol.x.reshape(1, 28*28)-xt[k].reshape(1, 28*28))
            np.savetxt(str(yt[k]) +'.csv', R, delimiter=",")
            N=N+1
        # sol=Clases.BlackBox(['saved_model/my_model_LR_BB.h5',(x_train[-1000:-1], y_train[-1000:-1]),(1,28*28)])
        r=r+1



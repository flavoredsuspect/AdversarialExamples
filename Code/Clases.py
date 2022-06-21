import tensorflow as tf
import numpy as np
import scipy.optimize
import scipy.stats
from numba import jit
n = 28


# Class to perform WhiteBox Attack
class WhiteBox:
    def __init__(self, x, params):

        # The original shape is saved to later perdform the call of the oracle, the input is reshaped as vector
        # xo is the original image and xp the perturbed one

        self.xo = x.reshape(np.prod(x.shape))
        params.append(x.shape)
        self.params = params
        print("Please, provide the following parameters."
              "\n >Target: [Targeted,Untargeted] "
              "\n >Type: [WhiteBox,BlackBox]"
              "\n >Directory of the model e.g: C:/Users/ted/Desktop/model.h5"
              "\n >Algorithm: [Lagrange Method (Write: LM),"
              "Iterative Fast Gradient Sign Method (Write:IFGSM),"
              "One-Pixel (Write:OP)]"
              "\n write alpha"
              )

        # Here the model is pulled apart of the last layer to obtain Z function later used in loss

        self.model = tf.keras.models.load_model(params[1])
        self.logit = tf.keras.models.Sequential()
        for layer in self.logit.layers[:-1]:
            self.logit.add(layer)

        if params[2] == "LM":
                print("Currently programmed for CNN & LR")
                self.xp = self.LM(self.xo,params)
        elif params[2] == "IFGSM":
                print("Currently programmed for LR")
                self.xp = self.IFGSM(self.xo)
        elif params[2] == "OP":
                self.xp = self.OP(self.xo)

    #xo es un vector !!!!

    def LM(self, xo, params):
        if params[3] == "NN":
            # alpha=10
            # The optimization process should be formulated to run in GPU

            if params[0] == "Untargeted":
                sol=scipy.optimize.minimize(self.Of_1,
                                                x0 = np.random.random(xo.shape[0]),
                                                args = (xo,params[4]),bounds=[(0,255) for i in xo],
                                                method = 'L-BFGS-B',
                                                options = {"eps": 1e-10, "disp": True}
                                                )
                print(sol)
                return(sol)

            elif params[0]=="Targeted":
                # alpha=0.1
                # No bounds are here needed since the formulation of Carlini makes a variable change

                print("Label, k")
                l=int(input())
                k=float(input())
                sol = scipy.optimize.minimize(self.Of_2,
                                              x0=np.random.random(xo.shape[0]),
                                              args=(xo, l, params[4], k),
                                              method='BFGS',
                                              options={"eps": 1e-2, "maxiter": 5, "disp": True}
                                              )
                print(sol)
                return (sol)

    def IFGSM(self, x):
        print("Not yet implemented")
    def OP(self, x):
        print("Not yet implemented")

    def Of_1(self, xp, xo, alpha):
        # The shape of the input fot the oracle needs t be formatted in the original shape
        # The scale for the left and right term of Of is non compensated and should be adjusted bu alpha

        cc = tf.keras.losses.CategoricalCrossentropy()
        return alpha*(np.power(np.linalg.norm(xo-xp),2))-cc(np.round(self.model.predict(xo.reshape(self.params[-1]))),
                                                       self.model.predict(xp.reshape(self.params[-1]))).numpy()

    def f(self, x , l, k):
        # A change to list for the ease of pop

        p = self.model.predict(x.reshape(self.params[-1]))[0].tolist()
        r = p.pop(l-1)
        y = max(p)-r
        return max(y,-k)

    def Of_2(self, xp, xo, l, alpha, k):

        return np.linalg.norm(0.5*(np.tanh(xp)+1)-xo) + alpha*self.f(0.5*(np.tanh(xp)+1), l, k)


class BlackBox:
    def __init__(self, params):
        # Here danger with the data format, tuple (1st index) of array (scn index)

        self.oracle=tf.keras.models.load_model(params[0])
        self.data=params[1]

        # The initial shape of input is stored to later be feeded into the oracle, the input is reshaped as vector


        self.params = params

        self.ro=9               # Number of augment iterations
        self.k=400             # Number of subset for RS, the dataset grows n*2^sigma+k*(p-sigma) here sigma=0
        self.alpha=0.1          # Step size
        self.params=params
        period=3

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        self.history=list()

        for i in range(0,self.ro):
            # The model is trained over the dataset since each iteration it is augmented

            self.history.append(self.model.fit(
                self.data[0],
                self.data[1],
                epochs=10,
                batch_size=128)
            )
            self.model.save('saved_model/my_model_subs_'+str(i)+'.h5')

            self.augment()
            # Periodical step size
            self.alpha= self.alpha*np.power(-1,np.floor(self.ro/period))

        self.history.append(self.model.fit(
            self.data[0],
            self.data[1],
            epochs=10,
            batch_size=128)
        )
        self.model.save('saved_model/my_model_subs.h5')

    def augment(self):
        N=len(self.data[0])

        # The tuple assignation needs to be done at once, also since data[0] is made from [array], to combine
        # the previous dataset and the new one, this assignation is done. When oracle predicts it also returns an
        # object and entry 0 is selected to combine. The round function is to assign an integer class.

        for j in range(0,self.k):
            print(j)
            self.data=(np.r_[self.data[0],[self.data[0][j]+ self.alpha*np.sign(self.jacobian(self.data[0][j],self.oracle.predict(self.data[0][j].reshape(self.params[-1]))[0]))]]
                       ,np.r_[self.data[1],[np.round(self.oracle.predict(self.data[0][-1].reshape(self.params[-1]))[0])]])

        for j in range(self.k,N+self.k):
            print(j)
            r=np.random.randint(0,j-1)
            if r<self.k:
                self.data[0][N+r]=self.data[0][j]+ self.alpha*np.sign(self.jacobian(self.data[0][j],self.oracle.predict(self.data[0][j].reshape(self.params[-1]))[0]))
                self.data[1][N+r] =  np.round(self.oracle.predict(self.data[0][-1].reshape(self.params[-1]))[0])

    def jacobian(self,x,p):
        n = np.prod(x.shape)
        xn=x.reshape(n)
        y=self.model.get_weights()[0]
        D_loss_D_scores = np.zeros((n,10))
        w=list()

        for j in range(0,10):
            w.append([x[j] for x in y])
        expsum=np.sum([np.exp(np.dot(wj,xn)) for wj in w])

        for i in range(0,n):  # for each training sample
            for j in range(0,10):  # for each class
                D_loss_D_scores[i,j]=(w[j][i]*np.exp(np.dot(w[j],xn))-np.sum([wj[i]*np.exp(np.dot(wj,xn)) for wj in w]))/expsum

        return np.matmul(D_loss_D_scores,p).reshape(x.shape)
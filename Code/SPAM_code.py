from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import text_to_image

import numpy as np
import pandas as pd
import dataframe_image as dfi

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import string
from gensim.models import Word2Vec

import time

import matplotlib.pyplot  as plt
import seaborn as sns

import numpy  as np
import pandas  as pd
import functools
import pickle

import tensorflow as tf
from tensorflow.python.platform  import  flags

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, Flatten
from keras import Model
from keras.optimizers  import  RMSprop , Adam
from keras.layers import Input

from cleverhans.attacks  import FastGradientMethod, SaliencyMapMethod
from cleverhans.utils_tf  import  model_train , model_eval , batch_eval, model_argmax
from cleverhans.attacks_tf  import  jacobian_graph
from cleverhans.utils  import  other_classes
from cleverhans.model import Model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import initializers
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport

from sklearn.preprocessing import  LabelEncoder , MinMaxScaler
from sklearn.multiclass import  OneVsRestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier , VotingClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import  LabelEncoder , MinMaxScaler
from sklearn.svm  import SVC , LinearSVC

from IPython.display import display
stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
df = pd.read_csv('spam.csv', encoding='latin-1')


df['patterns'] = df['sms_text'].apply(lambda x:' '.join(x.lower() for x in x.split()))
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
df['patterns']= df['patterns'].str.replace('[^\w\s]','')
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop_words))
df['patterns'] = df['patterns'].apply(lambda x: " ".join([wordnet_lemmatizer.lemmatize(word) for word in x.split()]))
df['patterns'] = df.apply(lambda row: nltk.word_tokenize(row['patterns']), axis=1)

display(df)

size = 100
window = 3
min_count = 1
workers = 3
sg = 1

##word2vect captures important information of each word in its context and codifies it
start_time = time.time()
tokens = pd.Series(df['patterns']).values
# Train the Word2Vec Model
w2v_model = Word2Vec(tokens, min_count = min_count, size = size, workers = workers, window = window, sg = sg)
print("Time taken to train word2vec model: " + str(time.time() - start_time))

word2vec_model_file = 'word2vec_' + str(size) + '.model'
w2v_model.save(word2vec_model_file)

##LOad the model and count words
sg_w2v_model = Word2Vec.load(word2vec_model_file)

# Total number of the words
print("Total number of words")
print(len(sg_w2v_model.wv.vocab))

##Save word2vec results into csv.
word2vec_filename = 'all_review_word2vec.csv'

with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in df.iterrows():
        model_vector = (np.mean([sg_w2v_model[token] for token in row['patterns']], axis=0)).tolist()

        if index == 0:
            header = ",".join(str(ele) for ele in range(size))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:
            line1 = ",".join([str(vector_element) for vector_element in model_vector])
        else:
            line1 = ",".join([str(0) for i in range(size)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')

##Read file and encode labels

word2vec_df = pd.read_csv(word2vec_filename)
word2vec_df['label'] = df['label']

# Encode labels
word2vec_df = pd.get_dummies(word2vec_df, columns=['label'])

display(word2vec_df.head(10))

headers = list(word2vec_df)
headers.remove('label_not_spam')
headers.remove('label_spam')

X = np.array(word2vec_df[headers].values.tolist())
y = np.array(word2vec_df[['label_not_spam', 'label_spam']].values.tolist())

##Split training and testing 70-30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("Value counts for training \n")
print(y_train[:, 0].size)
print("\n")
print("Value counts for testing \n")
print(y_test[:, 0].size)

###########INJECTION#########

a=df['patterns']
l=[len(i) for i in a]
L=max(l)
XX=np.zeros((5566,80,size))
YY=np.zeros((5566,2))
j=0
k=0
y = np.array(word2vec_df[['label_not_spam', 'label_spam']].values.tolist())

for i in df['patterns']:
    if len(i)!=0:
        h = w2v_model[i]
        b = np.zeros((80, size))
        b[0:h.shape[0], :] = h
        XX[j]=b
        YY[j]=y[k]
        j=j+1
    k=k+1

XX_train, XX_test, YY_train, YY_test= train_test_split(XX, YY, test_size=0.3)

model2 = Sequential()
layers = [
    Flatten(),
    Dense(256, activation='relu',input_shape=(L*size,)),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(2),
    Activation('softmax')
    ]
for l in layers:
    model2.add(l)

model2.compile(
    optimizer=Adam(0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history=model2.fit(
    XX_train,
    YY_train,
    epochs=5,
    batch_size=256,
    validation_data=(XX_test,YY_test)
)






##Train decision tree classifier

def classify(X_train, y_train, X_test, y_test):
    # Initialise Decision Tree
    clf = DecisionTreeClassifier()
    # Fit model
    model = clf.fit(X_train, y_train)
    # Predict testing target labels
    prediction = model.predict(X_test)

    return prediction


print(classification_report(y_test, classify(X_train, y_train, X_test, y_test)))

##Train a NN to classify
nb_epochs=1
batch_size=256
learning_rate=0.1
nb_classes=y_train.shape[1]
source_samples=X_train.shape[1]

def mlp_model(input_shape, input_ph=None, logits=False):
    # """145Generate a MultiLayer  Perceptron  model146"""
    model = Sequential()

    layers = [
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(nb_classes),
    ]

    for l in layers:
        model.add(l)

    if logits:
        logit_tensor = model(input_ph)

    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    if logits:
        return model, logit_tensor
    return model


model1 = Sequential()
layers = [
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(nb_classes),
    Activation('softmax')
    ]
for l in layers:
    model1.add(l)

model1.compile(
    optimizer=Adam(0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history=model1.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=256,
    validation_data=(X_test,y_test)
)


###################################################################################################################





def  evaluate():
    """164Model  evaluation  function165"""
    eval_params = {'batch_size': batch_size}
    train_acc = model_eval(sess, XX_t, YY_t, predictions , X_train , y_train , args=eval_params)
    test_acc = model_eval(sess, XX_t, YY_t, predictions , X_test , y_test , args=eval_params)
    print('Train acc: {:.2f} Test  acc: {:.2f} '.format(train_acc, test_acc))

plt.style.use('bmh')
flags = tf.app.flags
FLAGS = flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)



FLAGS = flags.FLAGS

####################################################################################################

# Generate  adversarial  samples  for  all  test  datapoints
source_samples = X_test.shape[0]
# Jacobian -based  Saliency  Map


results = np.zeros(XX.shape, dtype='i')
perturbations = np.zeros(XX.shape, dtype='f')

XX_t = tf.placeholder(tf.float32, shape=(None, 80, size))
YY_t = tf.placeholder(tf.float32, shape=(None, 2))


sess = tf.Session()
keras.backend.set_session(sess)

predictions = model2(XX_t)
init = tf.global_variables_initializer()
sess.run(init)


grads = jacobian_graph(predictions, XX_t,2)

X_adv = np.zeros(XX_test.shape)

wrap = KerasModelWrapper(model2)

# Loop over the samples we want to perturb into adversarial examples


samples_to_perturb = np.where(YY_test[:, 1] == 1)[0]  # only malicious
nb_classes1 = 2  # malicious or benign
nb_classes = 2

def model_pred(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)

    print(probabilities, "************")

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


def generate_adv_samples(samples_to_perturb, jsma_params):
    adversarial_samples = []
    samples_perturbed_idxs = []

    for i, sample_ind in enumerate(samples_to_perturb):
        sample = XX_test[sample_ind: sample_ind + 1]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(YY_test[sample_ind]))
        target = 1 - current_class

        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, nb_classes1), dtype=np.float32)
        one_hot_target[0, target] = 1
        jsma_params['y_target'] = one_hot_target

        adv_x = jsma.generate_np(sample, **jsma_params)  # adversarial sample generated = adv_x
        adversarial_samples.append(adv_x)
        samples_perturbed_idxs.append(sample_ind)

        # Check if success was achieved
        adv_tgt = np.zeros((1, nb_classes))  # adversarial target = adv_tgt
        adv_tgt[:, target] = 1
        res = int(model_eval(sess, XX_t, YY_t, predictions, adv_x, adv_tgt, args={'batch_size': 1}))

        # Compute number of modified features
        adv_x_reshape = adv_x.reshape(-1)
        test_in_reshape = X_test[sample_ind].reshape(-1)
        nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
        percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

        # Update the arrays for later analysis
        #results[target, sample_ind] = res
        #perturbations[target, sample_ind] = percent_perturb

    malicious_targets = np.zeros((len(adversarial_samples), 2))
    malicious_targets[:, 1] = 1

    adversarial_samples = np.stack(adversarial_samples).squeeze()
    original_samples = XX_test[np.array(samples_perturbed_idxs)]

    return adversarial_samples

jsma = SaliencyMapMethod(wrap, sess=sess)
jsma_params = {'theta': 0.5, 'gamma': 0.5, 'clip_min': 0., 'clip_max': 1., 'y_target': None}
adversarial_samples = generate_adv_samples(samples_to_perturb, jsma_params)

pred=model2.predict(adversarial_samples)
lab=YY_test[samples_to_perturb]
r=[]
for i in range(pred.shape[0]):
    r.append(np.array_equal(lab[i],pred[i]))


def reverse(x):
    word = []
    for i in x[:]:
        word.append(w2v_model.wv.most_similar(positive=[i,])[0])
    return [i[0] for i in word if i[1]!=0.0]

total=[reverse(i) for i in adversarial_samples]
cand=[i for i in total if len(i)<=8]
cand[2]=cand[12]
cand=cand[0:10]
data=pd.DataFrame(cand)
dfi.export(data,'perturbed_sms.png')
text=[''.join([j+' ' for j in i]) for i in cand]
text2=text

text2[0]='If u want explicit sex on 1 sec at ring cost, careless, then kkcongratulation! 600998746'
text2[1]='You have important customer service announcement premier! call freephone 600998746'
text2[2]= 'dating service call box334sk38ch or webpage 146tf150p '
text2[3]='please call immediately urgent message waiting physic, kkcongratulation! 600998746'
text2[4]='httptms widelivecomindex wmlid820554ad0a1705572711firsttrueåác c ringtoneåá ffffuuuuuuu 146tf150p lmaonice'
text2[5]='cash prize claim call09050000327 on september webpage'
text2[6]='new voicemail please call perumbavoor webpage acid lmaonice '
text2[7]='freeringtonereply real gwr timeyou  '
text2[8]='lost 3pound to help stalking webpage '
text2[9]='received mobile content enjoy operate webpage wrkin'

pd.set_option("display.max_colwidth", None)
data=pd.DataFrame(text2)
dfi.export(data,'perturbed_sms_text.png')

original=[]
original_sms=[]
for i in cand:
   com=[]
   for j in df['patterns']:
       com.append(np.sum([1 for k in j if k in i]))

   original.append(df['patterns'][com.index(max(com))])
   original_sms.append(df['sms_text'][com.index(max(com))])

#original[10]=df['patterns'][1776]
#original_sms[10]=df['sms_text'][1776]
data=pd.DataFrame(original)
pd.set_option("display.max_colwidth", None)
data2=pd.DataFrame(original_sms)
dfi.export(data2,'original_sms_text.png')
dfi.export(data,'original_sms.png')


gamma = []
theta = []

import itertools

for i in range(1, 10):
    gamma.append(i / 10)
    theta.append(i / 10)

combinations = list(itertools.product(gamma, theta))

jsma = SaliencyMapMethod(wrap, sess=sess)

final_results = []

for i in combinations:
    jsma_params = {'theta': i[1], 'gamma': i[0], 'clip_min': 0., 'clip_max': 1., 'y_target': None}
    adversarial_samples = generate_adv_samples(samples_to_perturb, jsma_params)
    adv_test = pd.DataFrame(adversarial_samples, columns=headers)

    adv_test['label_not_spam'] = 0
    adv_test['label_spam'] = 1

    test = pd.DataFrame(X_test, columns=headers)
    test['label_not_spam'] = y_test[:, 0]
    test['label_spam'] = y_test[:, 1]

    not_spam = test[test['label_not_spam'] == 1]

    joined = not_spam.append(adv_test, ignore_index=True)

    X_test_adv = np.array(joined[headers])
    y_test_adv = np.array(joined[['label_not_spam', 'label_spam']])

    final_results.append(f1_score(YY_test, model2.predict(X_test_adv), average='weighted'))

results = pd.DataFrame(combinations, columns=['Gamma', 'Theta'])
results['f1_score'] = final_results
display(results)

heatmap1_data = pd.pivot_table(results, values='f1_score', index=['Gamma'], columns='Theta')

fig, ax = plt.subplots(figsize=(20, 14))
ax = sns.heatmap(heatmap1_data, annot=True, ax=ax, cmap="YlGnBu")
ax.invert_yaxis()
plt.savefig('heatmap.png')
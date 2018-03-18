import numpy as np
import math
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import optimizers

def nn_model():
    input_data = Input(shape=[1])
    layer1 = Dense(10, activation='relu')(input_data)
    layer2 = Dense(10, activation='relu')(layer1)
    layer3 = Dense(10, activation='relu')(layer2)
    layer4 = Dense(10, activation='relu')(layer3)
    layer5 = Dense(10, activation='relu')(layer4)
    layer6 = Dense(10, activation='relu')(layer5)
    output = Dense(1, activation='tanh')(layer6)
    model = keras.models.Model(input_data, output)
    #sgd = optimizers.SGD(lr=0.07, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def normalize(X_all, X_test):
    # Feature normalizaion with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test)
    #mu = np.tile(mu, (X_train_test.shape[0], 1))
    #sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

## main program

model = nn_model()
xx2 = np.linspace(0.01,1,400000)
xx = np.random.uniform(low=0.01, high=1.0, size=400000)

yy_ = np.sign(np.sin(5*np.pi*xx))
yy2_ = np.sign(np.sin(5*np.pi*xx2))
## shuffle data
indices = np.arange(xx.shape[0])
np.random.shuffle(indices)
train = xx[indices]
label = yy_[indices]
# split data to train and val

nb_validation_samples = int(0.1 * train.shape[0])
train_ = train[nb_validation_samples:]
train_val = train[0:nb_validation_samples]


label_ = label[nb_validation_samples:]
label_val = label[0:nb_validation_samples]

## Normalizaion
#train_, train_val = normalize(train_, train_val)


model.fit(train_, label_, validation_data=(train_val, label_val), epochs=20, batch_size=512)

result = model.predict(xx2)

#plt.plot(xx, yy_)
plt.plot(xx2, yy2_, xx2, result)
plt.show()


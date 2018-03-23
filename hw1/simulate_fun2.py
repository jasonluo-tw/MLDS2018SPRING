import numpy as np
import math
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.initializers import Constant
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import optimizers

sgd = optimizers.SGD(lr=0.07, decay=1e-6, momentum=0.95, nesterov=True)

def deep_model():
    input_data = Input(shape=[1])
    layer1 = Dense(10, activation='relu', use_bias=True)(input_data)
    layer2 = Dense(10, activation='relu', use_bias=True)(layer1)
    layer3 = Dense(10, activation='relu', use_bias=True)(layer2)
    layer4 = Dense(10, activation='relu', use_bias=True)(layer3)
    layer5 = Dense(10, activation='relu', use_bias=True)(layer4)
    layer6 = Dense(10, activation='relu', use_bias=True)(layer5)
    layer7 = Dense(10, activation='relu', use_bias=True)(layer6)
    output = Dense(1, activation='linear', use_bias=True)(layer7)
    model = keras.models.Model(input_data, output)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def shallow_model():
    input_data = Input(shape=[1])
    layer1 = Dense(230, activation='relu', use_bias=True)(input_data)
    output = Dense(1, activation='linear', use_bias=True)(layer1)
    model = keras.models.Model(input_data, output)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def middle_model():
    input_data = Input(shape=[1])
    layer1 = Dense(16, activation='relu', use_bias=True)(input_data)
    layer2 = Dense(12, activation='relu', use_bias=True)(layer1)
    layer3 = Dense(13, activation='relu', use_bias=True)(layer2)
    layer4 = Dense(19, activation='relu', use_bias=True)(layer3)
    output = Dense(1, activation='linear', use_bias=True)(layer4)
    model = keras.models.Model(input_data, output)
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
epochs = 100

model = deep_model()
model2 = middle_model()
model3 = shallow_model()

xx = np.linspace(0.01,1,300000)

#xx = np.random.uniform(low=0.01, high=1.0, size=400000)
#yy2_ = np.sign(np.sin(5 * np.pi * xx2))

yy_ = np.sign(np.sin(5 *np.pi * xx))

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

model.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=512)
model2.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=512)
model3.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=512)

result = model.predict(xx)
result2 = model2.predict(xx)
result3 = model3.predict(xx)

plt.plot(xx, yy_, xx, result, xx, result2, xx, result3)
plt.legend(('True','deep model','middle model','shallow model'), loc='upper right')
plt.show()
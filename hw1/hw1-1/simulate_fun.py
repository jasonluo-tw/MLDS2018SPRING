import numpy as np
import math
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import optimizers

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

## main program
epochs = 15000

model = deep_model()
model2 = middle_model()
model3 = shallow_model()

xx = np.linspace(0.01,1,50000)
yy_ = np.sin(20*np.pi*xx) / (2*np.pi*xx + 0.5) * np.tan(2*np.pi*xx)

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

h_deep = model.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=1024, verbose=2)
h_middle = model2.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=1024, verbose=2)
h_shallow = model3.fit(train_, label_, validation_data=(train_val, label_val), epochs=epochs, batch_size=1024, verbose=2)

result = model.predict(xx)
result2 = model2.predict(xx)
result3 = model3.predict(xx)

plt.figure()
plt.plot(xx, yy_, xx, result, xx, result2, xx, result3)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\frac{sin(20 \pi x)}{2 \pi x + 0.5}*tan(2 * \pi x)$')
plt.legend(('True','deep model','middle model','shallow model'))
plt.savefig('fun1.png')

plt.figure()
x_epo = np.linspace(1, epochs, epochs)
plt.semilogy(x_epo, h_deep.history['loss'], 'orange')
plt.semilogy(x_epo, h_middle.history['loss'], 'green')
plt.semilogy(x_epo, h_shallow.history['loss'], 'red')
plt.xlabel("epochs")
plt.ylabel("loss (mse)")
plt.title('loss')
plt.legend(('deep model','middle model','shallow model'), loc='upper right')

plt.savefig("fun1_loss.png")

#plt.show()

import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras import regularizers


## define model
def nn_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())
    
    ## DNN
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    
    print(model.summary())
    return model

## plot_pics
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num >25: num = 25
    for i in range(num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx])
        title = str(i)+ ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()

### main program
## Declare variables
batch_ = 300

num_classes = 10
epo = 50

## Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

#plot_images_labels_prediction(x_train, y_train, [], 0)

## preprocess
## /255.
x_train = x_train / 255.
x_test = x_test / 255.

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = nn_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## call back function
call = EarlyStopping(monitor='val_acc', min_delta=0,patience=20)

### fit
train_history = model.fit(x_train, y_train, validation_split=0.2, epochs=epo, 
        batch_size=batch_, shuffle=True, callbacks=[call])


### evaluate model
scores = model.evaluate(x_test, y_test)
print()
print('accuracy=', scores[1])

### save model
from keras.models import load_model
model.save('Cifar_10.h5')

### save training history
import csv
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
text = open('./cifar_loss_acc.csv','w')
s = csv.writer(text,delimiter=',',lineterminator='\n')
for i in range(len(acc)):
        s.writerow([acc[i],val_acc[i],loss[i],val_loss[i]])

text.close()


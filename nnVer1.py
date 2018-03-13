from keras.models import Sequential
from keras.layers import Dense, Activation
import hold
import numpy as np

def train():
    model = Sequential()
    model.add(Dense(1, input_dim = 2))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim = 2))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim = 2))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim = 2))
    model.add(Activation('relu'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    data = np.asarray(hold.geti())
    labels = np.asarray(hold.geto())
    model.fit(data, labels, nb_epoch=100,verbose=1, batch_size=32)
    return model
model = train()
model.save('melanomaModel.h5')

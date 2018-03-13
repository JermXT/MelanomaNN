from keras.models import Sequential
from keras.layers import Dense, Activation
import hold
model = Sequential()
model.add(Dense(32, input_dim = 2))
model.add(Activation('relu'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
import numpy as np

#data = np.random.random((100, 2))
#labels = np.random.randint(2, size=(1000, 1))
data = np.asarray(hold.geti())
labels = np.asarray(hold.geto())
model.fit(data, labels, epochs=10,verbose=1, batch_size=32)

from keras.models import Sequential
from keras.layers import Dense, Activation
import hold
import numpy as np
from keras.models import load_model
#model = load_model('melanomaModel.h5')
def run(arr):
	model = load_model('melanomaModel.h5')
        data = np.asarray(arr)
        return model.predict(data,batch_size=1, verbose=0)


#print np.asarray(hold.test())
#run(np.asarray(hold.test()),model)

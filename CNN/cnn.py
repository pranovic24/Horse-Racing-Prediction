import keras
from keras.models import Sequential
from keras import models, layers
from keras.utils import np_utils

import numpy as np
import ast

def fetch_dataset(fileName):
	with open(fileName, 'r') as f:
	    resList = ast.literal_eval(f.read())
	return resList

# Read data
train_data = fetch_dataset("./preprocessed/"+"final_train.list")
train_labels = fetch_dataset("./preprocessed/"+"final_train_label.list")
test_data = fetch_dataset("./preprocessed/"+"final_test.list")
test_labels = fetch_dataset("./preprocessed/"+"final_test_label.list")

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

train_no = 2827 #2984
test_no = 315 #158

# Reshape the dataset into 3D array
train_data = train_data.reshape(train_no, 22,14)
train_labels = train_labels.reshape(train_no, 14)
test_data = test_data.reshape(test_no, 22,14)
test_labels = test_labels.reshape(test_no, 14)

#Instantiate an empty model
model = Sequential()

# C1 Convolutional Layer
model.add(layers.Conv1D(100, kernel_size=(2), strides=(1), activation='tanh', input_shape=(22,14), padding="valid"))

# S2 Pooling Layer
model.add(layers.AveragePooling1D(pool_size=(3), strides=(1), padding='valid'))

# C1 Convolutional Layer
model.add(layers.Conv1D(60, kernel_size=(5), strides=(1), activation='tanh', input_shape=(22,14), padding="same"))

# S2 Pooling Layer
model.add(layers.AveragePooling1D(pool_size=(3), strides=(1), padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv1D(40, kernel_size=(7), strides=(1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(layers.AveragePooling1D(pool_size=(3), strides=(1), padding='valid'))

#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC5 Fully Connected Layer
model.add(layers.Dense(200, activation='tanh'))

# FC6 Fully Connected Layer
model.add(layers.Dense(70, activation='tanh'))

#Output Layer with softmax activation
model.add(layers.Dense(14, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=["categorical_accuracy"])

model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='cnn_horse_rating.png', show_shapes=True, show_layer_names=True)

hist = model.fit(x=train_data,y=train_labels, epochs=40, batch_size=64, validation_data=(test_data, test_labels), verbose=1) 

test_score = model.evaluate(test_data, test_labels)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100)) 




from keras.models import Sequential
import random
import numpy as np
import cv2
import os
from tqdm import tqdm
from keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras import optimizers

path='train'
image_size = 50

batch_size = 64
n_epochs = 500
n_classes = 2

training_data = []
shuffled_data = []

def get_label_encoded(label):
	if label == 'PNEUMONIA' : return [1,0]
	elif label == 'NORMAL' : return [0,1]
	

def prep_data(path):
	path_disease = 'PNEUMONIA'
	path_normal = 'NORMAL'
	path_1=os.path.join(path,path_disease)
	path_2=os.path.join(path,path_normal)

	for img1 in tqdm(os.listdir(path_1)):
		image1 = os.path.join(path_1,img1)
		im1=cv2.resize(cv2.imread(image1,cv2.IMREAD_GRAYSCALE),(image_size,image_size))
		label_encoded1 = get_label_encoded(path_disease)
		training_data.append([np.array(im1),np.array(label_encoded1)])

	for img2 in tqdm(os.listdir(path_2)):
		image2 = os.path.join(path_2,img2)
		im2=cv2.resize(cv2.imread(image2,cv2.IMREAD_GRAYSCALE),(image_size,image_size))
		label_encoded2 = get_label_encoded(path_normal)
		training_data.append([np.array(im2),np.array(label_encoded2)])

	random.shuffle(training_data)

	return training_data
	
def train_test_split(training_data):
	train_data = training_data[:-200]
	test_data = training_data[-200:]

	train_X = np.array([i[0] for i in train_data]).reshape(-1,image_size,image_size,1)
	train_Y = np.array([i[1] for i in train_data])

	test_X = np.array([i[0] for i in test_data]).reshape(-1,image_size,image_size,1)
	test_Y = np.array([i[1] for i in test_data])

	return train_X,train_Y,test_X,test_Y

def Conv_net(input_shape):
	model = Sequential()
	
	model.add(Conv2D(10,(10,10),padding = 'same',activation ='relu',input_shape=input_shape))
	model.add(Conv2D(5,(5,5),padding = 'same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(10,(10,10),padding = 'same',activation ='relu',input_shape=input_shape))
	model.add(Conv2D(5,(5,5),padding = 'same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(10,(10,10),padding = 'same',activation ='relu',input_shape=input_shape))
	model.add(Conv2D(5,(5,5),padding = 'same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(500,activation = 'relu'))
	model.add(Dropout(0.25))
	model.add(Dense(n_classes,activation = 'softmax'))

	return model

def train_network(model,train_X,train_Y,test_X,test_Y):
	
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=6, min_lr=0.00001,mode='auto',min_delta=0.0001,verbose=1)

	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=1, mode='auto',restore_best_weights=True)

	model.compile(loss='binary_crossentropy',optimizer ='adam',metrics = ['accuracy'])
	model.fit(train_X,train_Y,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(test_X,test_Y),callbacks=[reduce_lr,early_stop])
	score = model.evaluate(test_X,test_Y,verbose = 0)
	print("Test Loss = ",score[0])
	print("Test Accuracy = ",score[1])
	model.save('Saved_model/Pneumonia_Classifier.h5')
	

training_data = prep_data(path)
train_X,train_Y,test_X,test_Y = train_test_split(training_data)
model = Conv_net(train_X.shape[1:])
train_network(model,train_X,train_Y,test_X,test_Y)


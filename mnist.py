#import necessary packages
import tensorflow as tf 
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)

#import mnist dataset
from keras.datasets import mnist 

#import mnist data
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

"""
#Show image
def plot_img(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()
plot_img(x_train_image[0])
"""

#to see the prediction more convinent, def plot_images_prediction function
def plot_images_prediction(iamges, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(iamges[idx], cmap='binary')
        title = "label" + str(labels[idx])
        if len(prediction)>0 :
            title += ",prediction=" + str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()

plot_images_prediction(x_train_image,y_train_label,[],0,10)

#Data preprocess section
#Reshape the datasets as a one dimensional array
x_train_img_res = x_train_image.reshape(60000,784).astype('float32')
x_test_img_res = x_test_image.reshape(10000,784).astype('float32')

#Normalize the image 
x_train_img_res_normalized = x_train_img_res/255
x_test_img_res_normalized = x_test_img_res/255

#Doing one-hot encoding for labels
y_trainOnehot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)

#Mdoel Construction
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#Input layer 1: 256 units of neurons with 784 input dimensions. 
#Initializer: Normal distribution. 
#Activation function: relu. 
model.add(Dense(units = 256, input_dim=784, kernel_initializer='normal', activation='relu'))

#Input layer 2: 10 units of neurons with 256 input dimensions. 
#Initializer: Normal distribution. 
#Activation function: softmax. 
model.add(Dense(units = 10, kernel_initializer='normal', activation='softmax'))

#Model Compiling 
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
train_history = model.fit(x=x_train_img_res_normalized, y=y_trainOnehot, 
validation_split=0.2, epochs=10, batch_size=250, verbose=2)
#Print model summary  
print(model.summary())

#Define a function to show training history
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validaiton'],loc='upper left')
    plt.show()

#Show the result of training 
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')

#Scoring the accuracy of model by test dataset.
score = model.evaluate(x_test_img_res_normalized, y_testOneHot)
print()
print("Accuracy of model is", score[1])

#Prediction 
prediction = model.predict_classes(x_test_img_res)
print(prediction)

plot_images_prediction(x_test_image,y_test_label,prediction,idx=1,num=5)

#Display confusion matrix
pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])

#Display true value and predict value 
#df = pd.DataFrame({'label':y_test_label,'predict':prediction})
#df[:10]

#Find the result that true value is x but pred value is y (x != y)
#df[(df.label==5)&(df.predict==3)]


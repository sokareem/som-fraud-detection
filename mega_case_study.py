# Mega Case Study - Make a Hybrid Deep Learning Model


#Part 1 - Identify the Frauds with the Self-Organizing Map
#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

#split our dataset
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10,y = 10,input_len= 15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
#create markers to show approval status
markers = ['o','s']
colors = ['r','g']
for i, p in enumerate(x):
  #get winning node
  win = som.winner(p)
  plot(win[0] + 0.5,win[1] + 0.5,markers[y[i]], markeredgecolor = colors[y[i]],markerfacecolor = 'None',markersize = 10, markeredgewidth = 2)

show()

# Finding the frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(6,6)], mappings[(5,7)]),axis = 0)
#Inverse feature scaling
frauds = sc.inverse_transform(frauds)
print("frauds = "+ str(frauds))



#Part 2 - Going from Unsupervised to Supervised Deep Learning

#Create the matrix of features
customers = dataset.iloc[:,1:].values


# Creating the dependent variables
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if(dataset.iloc[i,0] in frauds):
    is_fraud[i] = 1


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 Let's make the artificial neural network!
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
#create object of the sequential class
classifier = Sequential()

#Adding the input layer and first hidden layer with dropout
classifier.add(Dense(activation="relu", input_dim=15, units=2, kernel_initializer="uniform")) #12 input layers and 6 hidden layers
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) # use sigmoid activation function
#NOTE: If you have a dependent variable that has more than 2 categories use softmax function which is for one hot encoded dependent variable

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(customers,is_fraud,batch_size = 1, epochs = 3)#input, output,processing batch size,rounds of training

# Part 3 - Making the prediction and evaluating the model

#Predicting the probabilities of frauds
y_pred = classifier.predict(customers)


#append the Customer ID column and sort the probabilities
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()] #sorts numpy array by column of index 1

print("predicted probabilities of frauds = "+str(y_pred))
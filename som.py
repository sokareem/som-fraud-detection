#Self Organizing Map - Sinmisola Kareem

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
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]),axis = 0)
#Inverse feature scaling
frauds = sc.inverse_transform(frauds)
print("frauds = "+ str(frauds))


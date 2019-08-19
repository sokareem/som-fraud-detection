
# som-fraud-detection
Using a Self Organizing Map, we will create a map that looks for the outliers in the map indicating potential fraudulent users and try to find their Customer_ID

# Training the SOM
Step 1: We start with a dataset composed of n independent variables/attributes

Step 2: We create a grid composed of nodes, each having a weight vector of n elements

Step 3: We randomly initialize values of weight vectors to small numbers close to 0 (but not 0) 

Step 4: Select one random observation point from the dataset

Step 5: Compute the Euclidean distances from this point to the different neurons in the network

Step 6: Select the neuron that has the minimum distance to the point. This neuron is the winning neuron/node

Step 7: Update the weights of the winning node to move it closer to the point.

Step 8: Using a Gaussian neighborhood function of mean the winning node, also update the weights of the winning node neighbors to move them closer to the point. The neighborhood radius is the sigma in the Gaussian function.

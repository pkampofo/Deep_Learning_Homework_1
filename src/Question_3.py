#Import needed libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
N = 100 #number of data points
poly_data = pd.read_csv('../datasets/Question_3_dataset/data_ex1.csv')

#------------------------------------------------------#
#  	   A) Deducing minimum (a*, b*) numerically        #
#------------------------------------------------------#

x = np.empty([N, 2]) 		 # create 100 by 2 matrix to hold x
x[:,0] = 1 					 # fill the matrix with 1s in first column
x[:,1] = poly_data['x']		 # put poly_data['x'] in second column
y = np.array(poly_data['y']) # create 100 by 1 matrix to hold y
#Find the min, (a*, b*)
theta_min = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y) 
print('(a*, b*) is: (' + str(theta_min[0]) + ', ' + str(theta_min[1]) + ')\n')

#------------------------------------------------------#
#  	   B) i) Gradient Descent Implementation           #
#------------------------------------------------------#

learn_rate = 0.05
epochs = 2500
theta = np.array([1.8, 1.4]) # begin at (a0, b0) = (1.8, 1.4)

# Append norm of (a_n, b_n) - (a_*, b_*) to param_norm on every iteration
param_norm = [] # will be used in estimating convergence and plotting

for i in range(epochs):
    param_norm.append(np.linalg.norm(theta - theta_min))
    # Get predictions using current theta
    predictions = x.dot(theta)
    error_vector = y - predictions # Calculte error
    grad_vector = -2/N * x.T.dot(error_vector) # Calculate gradients 
    theta = theta - learn_rate*grad_vector # Modify theta

print("Gradient Descent algorithm:")
print("		(a, b) = (" + str(theta[0]) + ", " + str(theta[1]) + ")") # print best (a,b) found by GD.


#-------------------------------------------------------------------------#
#  	  B) ii) Rate of convergence of || (a_n, b_n) - (a_*, b_*) || to 0    #
#	                 for Gradient Descent algorithm              	  	  #
#-------------------------------------------------------------------------#

eps = 1e-7 # choose epsilon very close to 0
# find how many epochs it takes for || (a_n, b_n) - (a_*, b_*) || < 1e-7
epochs_taken = np.where(np.array(param_norm) < eps)[0][0]
distance_covered = param_norm[0] - eps # 'distance'
conv_rate = distance_covered / epochs_taken 
print("		||(a_n, b_n) - (a_*, b_*)|| --> 0 at average rate of: " + \
      str(conv_rate) + " per epoch.")

plt.figure(1)
plt.plot(param_norm) # plot the evolution of || (a_n, b_n) - (a_*, b_*) || against epochs.
plt.grid()
plt.xlabel("Epoch", fontsize=14);
plt.ylabel("Norm of  theta - theta_min", fontsize=14);
plt.title("Gradient Descent Algorithm evolution \n Convergence rate: " + str(conv_rate) + " per epoch");
plt.savefig("../results/Question_3_results/gradient_descent_convergence")


#------------------------------------------------------#
#  	   C) i) Momentum Implementation           		   #
#------------------------------------------------------#
N = 100
learn_rate = 0.05
gamma = 0.9
epochs = 500
theta = np.array([1.8, 1.4]) # begin at (a0, b0) = (1.8, 1.4)
v_old = 0; #start with v_n of 0

param_norm = [] # append norm of (a_n, b_n) - (a_*, b_*) to this on every iteration

for i in range(epochs):
    param_norm.append(np.linalg.norm(theta - theta_min))
    # Get predictions using current theta
    predictions = x.dot(theta)
    error_vector = y - predictions # Calculte error
    grad_vector = -2/N * x.T.dot(error_vector) # Calculate gradients
    v_new = gamma * v_old + learn_rate * grad_vector
    theta = theta - v_new # Modify theta
    
    #store v_new
    v_old = v_new

print("\nMomentum algorithm:")
print("		(a, b) = (" + str(theta[0]) + ", " + str(theta[1]) + ")") # print best (a,b) found by momentum

#-------------------------------------------------------------------------#
#  	  C) ii) Rate of convergence of || (a_n, b_n) - (a_*, b_*) || to 0    #
#	                 for Momentum algorithm              	  	          #
#-------------------------------------------------------------------------#

eps = 1e-7 # choose epsilon very close to 0
# find how many epochs it takes for || (a_n, b_n) - (a_*, b_*) || < 1e-7
epochs_taken = np.where(np.array(param_norm) < eps)[0][0]
distance_covered = param_norm[0] - eps # 'distance'
conv_rate = distance_covered / epochs_taken 
print("		||(a_n, b_n) - (a_*, b_*)|| --> 0 at average rate of: " + \
      str(conv_rate) + " per epoch.")

plt.figure(2)
plt.plot(param_norm) # plot the evolution of || (a_n, b_n) - (a_*, b_*) || against epoch.
plt.grid()
plt.xlabel("Epoch", fontsize=14);
plt.ylabel("Norm of  theta - theta_min", fontsize=14);
plt.title("Momentum algorithm evolution \n Convergence rate: " + str(conv_rate) + " per epoch");
plt.savefig("../results/Question_3_results/momentum_convergence")



#------------------------------------------------------#
#  	   C) iii) Nesterov Implementation           	   #
#------------------------------------------------------#
# Implement Nesterov algorithm
N = 100
learn_rate = 0.05
gamma = 0.9
epochs = 500
theta = np.array([1.8, 1.4])
v_old = 0; #start with v_n of 0

param_norm = [] # save norm of (a_n, b_n) - (a_*, b_*) on every iteration

for i in range(epochs):
    param_norm.append(np.linalg.norm(theta - theta_min))
    theta_mod = theta - gamma*v_old # sort of 'future' theta. This is what differentiates Nesterov.
    prediction_at_theta_mod = x.dot(theta_mod)
    error_vector_at_theta_mod = y - prediction_at_theta_mod
    
    grad_vector_at_theta_mod = -2/N * x.T.dot(error_vector_at_theta_mod) # Calculate gradients

    v_new = gamma * v_old + learn_rate * grad_vector_at_theta_mod
    theta = theta - v_new # Modify theta
    
    #store v_new
    v_old = v_new

print("\nNesterov algorithm:")
print("		(a, b) = (" + str(theta[0]) + ", " + str(theta[1]) + ")") # print best (a,b) found by Nesterov

#-------------------------------------------------------------------------#
#  	  B) iv) Rate of convergence of || (a_n, b_n) - (a_*, b_*) || to 0    #
#	                 for Nesterov algorithm              	  	          #
#-------------------------------------------------------------------------#

eps = 1e-7 # choose epsilon very close to 0
# find how many epochs it takes for || (a_n, b_n) - (a_*, b_*) || < 1e-7
epochs_taken = np.where(np.array(param_norm) < eps)[0][0]
distance_covered = param_norm[0] - eps # 'distance'
conv_rate = distance_covered / epochs_taken 
print("		||(a_n, b_n) - (a_*, b_*)|| --> 0 at average rate of: " + \
      str(conv_rate) + " per epoch.")

plt.figure(3)
plt.plot(param_norm) # plot the evolution of || (a_n, b_n) - (a_*, b_*) || against epoch.
plt.grid()
plt.xlabel("Epoch", fontsize=14);
plt.ylabel("Norm of  theta - theta_min", fontsize=14);
plt.title("Nesterov algorithm evolution \n Convergence rate: " + str(conv_rate) + " per epoch");
plt.savefig("../results/Question_3_results/nesterov_convergence")


#------------------------------------------------------#
#  	   D)  Other Gradient Descent Methods              #
#------------------------------------------------------#

import torch
import torch.nn as nn

myModel = nn.Linear(2, 1, bias=False) # Have already added 1 to each data point 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.05)
# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from matplotlib import rcParams

def find_polys_and_losses(x, y):
	#   Given datapoints x,y, this function returns predicted 
	#	polynomials of orders 0 to 12
    ks = range(13)
    predicted_polys = []
    losses = []

    for i in ks:   
        #full=True argument makes polyfit return 
        #more info than only coefficients
        sol = np.polyfit(x, y, i, full = True)   
        #Grab coefficients and sum of square errors of polynomial
        coeffs, sum_of_sq_errors = sol[:2] 
        #store polynomial, together with its loss
        predicted_polys.append(np.poly1d(coeffs))
        #divide by len(x) to get mean-square-error since polyfit returns only square-error
        losses.append(sum_of_sq_errors[0] / len(x))
   
    return predicted_polys, losses


def plot_predicted_polys(x, y, predicted_polys, dataset_label, save_to):
	#This function plots predicted polynomials along with 
	#corresponding dataset x,y
    rcParams['figure.figsize'] = 12,35
    x_new = np.linspace(np.min(x), np.max(x) + 0.1, 200)

    plt.figure()
    for i in range(13):
        y_new = predicted_polys[i](x_new)
    
        plt.subplot(7,2, i+1);
        plt.plot(x, y, 'o', label = dataset_label); #plot the data points
        plt.plot(x_new, y_new, 'r-', label='Predicted polynomial, k = ' + str(i));
        plt.grid(); plt.xlabel('x', fontsize=15); plt.ylabel('y', fontsize=15);
        plt.tight_layout(); plt.legend(loc='best');

    plt.savefig(save_to)


def plot_loss_against_k(save_to, losses, plot_title, y_label, losses2=None, plot_title2=None, y_label2=None):
    #plot loss as function of k. 
    #Use spline to make graph smooth
    rcParams['figure.figsize'] = 6.5,5.5
    
    ks = range(13)
    
    t, c, k = interpolate.splrep(ks, losses)
    spline = interpolate.BSpline(t, c, k)
    ks_new = np.linspace(0, 12, 200)
    plt.figure();
    plt.plot(ks_new, spline(ks_new), '-', label = y_label);
    
    if not losses2:
    	plt.plot(ks, losses, 'ro'); #Plot the dots only if only one loss is being plotted
    
    if losses2:
        t2, c2, k2 = interpolate.splrep(ks, losses2)
        spline2 = interpolate.BSpline(t2, c2, k2)
        plt.plot(ks_new, spline2(ks_new), '-', label = y_label2);

    plt.grid(); 
    if losses2:
        plt.legend(loc='best', fontsize=16); #show legend only if there are 2 losses being plotted together
    plt.xlabel('k', fontsize=16);
    plt.ylabel(y_label if not losses2 else 'Loss', fontsize=16);
    plt.title(plot_title, fontsize=16);
    plt.savefig(save_to) #save figure

# Load data
poly_data = pd.read_csv('../datasets/Question_1_dataset/data_ex1.csv')
x, y = poly_data['x'], poly_data['y']

#------------------------------------------------------#
#  	   A) Estimate polynomials with full dataset       #
#------------------------------------------------------#
# Predict polynomials and their losses
predicted_polys, losses = find_polys_and_losses(x, y) 
# Plot predicted polynomials along with dataset
plot_predicted_polys(x, y, predicted_polys, '100% dataset',
	'../results/Question_1_results/predicted_polys_100%_data.pdf')
# Plot loss as a function of k
plot_loss_against_k('../results/Question_1_results/loss_against_k', losses, 'Loss as a function of k\n (full dataset)', \
	'Loss')


#------------------------------------------------------#
#  	   B) i) Split data into train and test            #
#------------------------------------------------------#
# Goal: randomly choose 80% of data for train and 20% for test
dataset_count = poly_data.shape[0]
train_count = int(.8 * dataset_count)

#Ensure that same permutation is generated every time code is run for replicatable results
np.random.seed(0)

#Generate shuffled integers from 0 to 99. , 
shuffled_ints = np.random.permutation(dataset_count) 
train_indices = shuffled_ints[:train_count] #Choose first 80% for train
test_indices = shuffled_ints[train_count:] # Choose last 20% for test

x_train_data, y_train_data = x[train_indices], y[train_indices]
x_test_data, y_test_data = x[test_indices], y[test_indices]


#------------------------------------------------------------------#
#  	B) ii) Estimate polynomials using training set (80% of data)   #
#------------------------------------------------------------------#
predicted_polys_train, losses_train = find_polys_and_losses(
										x_train_data, y_train_data)
# Plot predicted polynomials
plot_predicted_polys(x_train_data, y_train_data, predicted_polys_train, 
		"Train dataset (80% of data)", '../results/Question_1_results/predicted_polys_80%_data.pdf')

# Do prediction on unseen test data and keep losses
losses_test = [] # Will append 

for poly in predicted_polys_train: # loop through predicted polys 
    predicted_ys = poly(x_test_data) # do prediction of y
    mse = np.mean((y_test_data - predicted_ys)**2) # calculate MSE
    losses_test.append(mse) # store MSE

# Plot losses_train and losses_test for all k
plot_loss_against_k('../results/Question_1_results/l_train_and_l_test_for_all_k', 
					losses_train, 'Loss_train and Loss_test against k', 'Loss_train',
                    losses_test, 'Loss_train and Loss_test against k', 'Loss_test')


#------------------------------------------------------------------#
#  				C) Order of polynomial P  						   #
#------------------------------------------------------------------#

# My guess is that the order of P is 2. 
#It's coefficients are: [0.39284984, -0.38975863, 0.82121564] with higher order coefficients first.
print("My guess is that the order of P is 2. \n \
	It's coefficients are: [0.39284984, -0.38975863, 0.82121564]\
	with higher order coefficients first.")
# The polynomial of order 2 predicted using 100% of the dataset.
print("\n " + str(predicted_polys[2])) 
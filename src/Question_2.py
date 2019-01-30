import matplotlib.pyplot as plt
#----------------------------------------------#
#  B) Evolution of average square distance as
#				function of d
#----------------------------------------------#
d_max = 31 # Plot for only dimension 1 to 50
squared_distances_mean = [d/6 for d in range(1, d_max)]

plt.figure()
plt.plot(range(1, d_max), squared_distances_mean, '-o')
plt.xlabel('Dimension, d', fontSize=15)
plt.ylabel('Average square distance, d/6', fontSize=15)
plt.title('Evolution of average square distance with dimension')
plt.savefig('../results/Question_2_results/average_square_distance_vs_d')

#----------------------------------------------#
#  C) Square Euclidean norm of similar images  #
#----------------------------------------------#
import numpy as np
from PIL import Image

image1 = np.array( Image.open('../datasets/Question_2_dataset/fox1.png') )
image2 = np.array( Image.open('../datasets/Question_2_dataset/fox2.png') ) 

x1 = image1/255
x2 = image2/255

# square Euclidean distance
sq_eucl_dist = np.sum((x1-x2)**2)

print("The square Euclidean distance of the two images is " + str(sq_eucl_dist))
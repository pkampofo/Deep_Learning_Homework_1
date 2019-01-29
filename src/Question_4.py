import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms #used in transforming PIL image to tensor

import numpy as numpy
import matplotlib.pyplot as plt 
import pandas as pd
import time

# store hyperparameters in a dictionary
P = {
	'no_of_epochs': 40,
	'learning_rate': 1e-4,
	'batch_size': 50
}

#----------------------------------------#
#           A) Initialization            #
#----------------------------------------#

# load FashionMNIST data
data_loc = '../datasets/Question_4_dataset'
train_data = datasets.FashionMNIST(data_loc, train=True, download=True, 
										transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(data_loc, train=False, download=True, 
										transform=transforms.ToTensor())
train_loader = DataLoader(train_data, shuffle=True, batch_size=P['batch_size'])
test_loader = DataLoader(test_data, shuffle=False, batch_size=P['batch_size'])

# Initiate linear classifier model
myModel = nn.Linear(28*28,10)
# Use cross entropy loss function
criterion = nn.CrossEntropyLoss()
# Use the Adam optimizer
optimizer = torch.optim.Adam(myModel.parameters(), lr=P['learning_rate'])
# Store useful statistics in DataFrame object
t0 = time.time()
N_train = len(train_data)   # number of images in the training set
N_test = len(test_data)     # number of images in the test set
df = pd.DataFrame(columns=('epoch', 'loss', 'accuracy_train', 'accuracy_test'))


#----------------------------------------#
#           B) training                  #
#----------------------------------------#

for epoch in range(P['no_of_epochs']):
    
    print('-- epoch '+str(epoch)) #print current epoch
    running_loss = 0.0
    accuracy_train = 0.0

    myModel.train()             # 'unfreeze' the parameters
    for X,y in train_loader:
        # init
        optimizer.zero_grad()
        # forward (prediction)
        N,_,nX,nY = X.size() # X size [N,1,nX,nY] with N size of batch
        score = myModel(X.view(N,nX*nY))
        # error
        loss = criterion(score, y)
        # one step optimizer
        loss.backward()
        optimizer.step()
        # estimate the loss and accuracy
        running_loss += loss.detach().numpy()
        accuracy_train += (score.argmax(dim=1) == y).sum().numpy()
 
    # Check and record statistics for this epoch
    myModel.eval()             # 'freeze' the parameters
    accuracy_test = 0.0

    for X,y in test_loader:
        # forward (prediction)
        N,_,nX,nY = X.size() # X size [N,1,nX,nY] with N size of batch
        score = myModel(X.view(N,nX*nY))
        # accuracy
        accuracy_test += (score.argmax(dim=1) == y).sum().numpy()
    
    # normalize
    accuracy_train /= N_train
    accuracy_test /= N_test
    average_loss = running_loss/N_train*P['batch_size']

    # print terminal
    print('    loss = '+str(average_loss))
    print('    accuracy (train/test) : '+str(accuracy_train)+' '+str(accuracy_test))
    # save
    df.loc[epoch] = [epoch, average_loss, accuracy_train, accuracy_test]


#----------------------------------------#
#           C) Plot results              #
#----------------------------------------#

# Print time elapsed in seconds
elapsed = time.time() - t0
print(' Total time (s) : '+str(elapsed))

# Plot evolution of the loss, train accuracy, and test accuracy
plt.figure(1);plt.clf()
plt.plot(df['epoch'],df['loss'],'-o')
plt.plot(df['epoch'],df['accuracy_train'],'-o')
plt.plot(df['epoch'],df['accuracy_test'],'-o')
plt.axis([0, P['no_of_epochs'], 0, 1])
plt.grid()
plt.xlabel('epochs')
plt.legend(['loss', 'train accuracy', 'test accuracy'])
plt.savefig('../results/Question_4_results/loss_and_accuracy_evolution')

# Save images of all templates
w = myModel.state_dict()['weight']

# Dictionary of class names
label_fashion = dict([(0,'T-shirt'),(1,'trouser'),(2,'pullover'),(3,'dress'),(4,'coat'),
(5,'sandal'),(6,'shirt'),(7,'sneaker'),(8,'bag'),(9,'boot')])

plt.figure(2)
for i in range(10):
	plt.subplot(3,4, (i+1));
	plt.imshow(w[i].view(28,28), cmap='gray')
	plt.title(label_fashion[i])
	#plt.colorbar(extend="both")
	#plt.show()
plt.tight_layout()
plt.savefig('../results/Question_4_results/templates')	
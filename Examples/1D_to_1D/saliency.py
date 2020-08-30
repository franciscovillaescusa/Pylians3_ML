import numpy as np
import sys,os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import data
import architecture


#################################### INPUT ##########################################
# data parameters
root_bn = '/mnt/ceph/users/fvillaescusa/CDS_2019/bottleneck_regression/bn_512_mean_new'
root_Pk = '/mnt/ceph/users/fvillaescusa/CDS_2019/bottleneck_regression/ps'
seed    = 1
realizations = 2000

# architecture parameters
h1 = 1000
h2 = 400
h3 = 100
dropout_rate = 0.0

# training parameters
batch_size = 1

# name of output files
name   = '3hd_250_250_250_0.0_3e-4'
fmodel = 'models/%s.pt'%name
#####################################################################################

# use GPUs if available
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU:',GPU)
print('Training on',device)
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
test_loader  = data.create_dataset('test', seed, realizations, root_bn, root_Pk, 
                                   batch_size)

# get the parameters of the trained model
#model = architecture.model_1hl(bins_SFRH, h1, 6, dropout_rate)
#model = architecture.model_2hl(bins_SFRH, h1, h2, 6, dropout_rate)
model = architecture.model_3hl(512+158, h1, h2, h3, 5, dropout_rate)
model.load_state_dict(torch.load(fmodel))
model.to(device=device)

# grab 1 element
model.eval()
for x,y in test_loader:
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        break

# allow input to have gradients
x.requires_grad_()

# get neural network prediction and compute loss
y_NN = model(x)
loss = criterion(y_NN, y)

# get the gradients of the network with respect to the output
y_NN.backward()

# do a loop 
for i in range(y_NN.shape[1]):
    #print(x.grad)
    #print(y_NN.shape)
    y_NN[0,i].backward(retain_graph=True) #backward first element of NN output
    saliency = (x.grad[0,0]).cpu().detach().numpy()
    print('Saliency =',saliency)

    #print(saliency.shape)
    #np.save('saliency_map_0_%d.npy'%i,saliency)
    x.grad[0,0].data.zero_()

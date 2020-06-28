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

################################### INPUT ############################################
# data parameters
root  = '/mnt/ceph/users/fvillaescusa/Ana/Gaussian_density_fields_3D'
f_in  = '%s/data/Gaussian_df_3D.npy'%root
f_out = '%s/data/A_values_3D.npy'%root
seed  = 10 #to split cubes between training, validation and testing sets

# architecture parameters
lr         = 1e-5
hidden     = 8
dr         = 0.0
wd         = 3e-2
epochs     = 100000
batch_size = 128

# output files
f_loss  = 'losses/loss_b_6x_%s_dr=0.0_wd=3e-2.txt'%hidden
f_model = 'models/model_b_6x_%s_dr=0.0_wd=3e-2.pt'%hidden
######################################################################################

# use GPUs if available
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s     ||    Training on %s\n'%(GPU,device))
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
train_loader = data.create_dataset_cubes('train', seed, f_in, f_out, batch_size, 
                                         verbose=True)
valid_loader = data.create_dataset_cubes('valid', seed, f_in, f_out, batch_size,
                                         verbose=True)
test_loader  = data.create_dataset_cubes('test',  seed, f_in, f_out, batch_size, 
                                         verbose=True)

# define the model
#model = architecture.model_a(hidden).to(device)
model = architecture.model_b(hidden).to(device)

# load best-models, if they exists
if os.path.exists(f_model):  model.load_state_dict(torch.load(f_model))

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)


# see if results for this model are available
if os.path.exists(f_loss):  
    dumb = np.loadtxt(f_loss, unpack=False)
    offset = int(dumb[:,0][-1]+1)
else:   offset = 0

# do a loop over all the epochs
min_loss = 1e8
for epoch in range(offset, offset+epochs):
    
    # training
    train_loss, num_points = 0.0, 0
    model.train()
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_NN = model(x)
        loss = criterion(y_NN, y)
        train_loss += (loss.cpu().item())*x.shape[0]
        num_points += x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/num_points

    # validation
    valid_loss, num_points = 0.0, 0
    model.eval()
    for x,y in valid_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_NN = model(x)
            loss = criterion(y_NN, y)
            valid_loss += (loss.cpu().item())*x.shape[0]
            num_points += x.shape[0]
    valid_loss = valid_loss/num_points

    # testing
    test_loss, num_points = 0.0, 0
    model.eval()
    for x,y in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_NN = model(x)
            loss = criterion(y_NN, y)
            test_loss += (loss.cpu().item())*x.shape[0]
            num_points += x.shape[0]
    test_loss = test_loss/num_points

    # verbose
    if valid_loss<min_loss:
        min_loss = valid_loss
        torch.save(model.state_dict(), f_model)
        print('Epoch %d: %.3e %.3e %.3e (saving)'%(epoch, train_loss, valid_loss, 
                                                   test_loss))
    else:
        print('Epoch %d: %.3e %.3e %.3e'%(epoch, train_loss, valid_loss, test_loss))

    # save losses
    f = open(f_loss, 'a')  
    f.write('%d %.3e %.3e %.3e\n'%(epoch, train_loss, valid_loss, test_loss))
    f.close()


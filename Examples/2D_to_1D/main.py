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
# images parameters
root  = '/mnt/ceph/users/camels/Results/Gaussian_fields/data'
#f_in  = '%s/Gaussian_maps_kpivot=0.3_discon.npy'%root 
#f_out = '%s/A_values_kpivot=0.3_discon.npy'%root
f_in  = '%s/Gaussian_maps_kpivot=0.3.npy'%root 
f_out = '%s/A_values_kpivot=0.3.npy'%root
seed  = 20 #to split images between training, validation and testing sets

# architecture parameters
lr         = 1e-5
hidden     = 16
dr         = 0.0
wd         = 3e-4
epochs     = 100000
batch_size = 128
model      = 'model_f'

# output files
root_out = '/mnt/ceph/users/camels/Results/Gaussian_fields'
#f_loss   = '%s/losses/loss_%s_%d_wd=%.1e_kpivot=0.3_discon_0.3variation.txt'%(root_out, model, hidden, wd)
#f_model  = '%s/models/%s_%d_wd=%.1e_kpivot=0.3_discon_0.3variation.pt'%(root_out, model, hidden, wd)
f_loss   = '%s/losses/loss_%s_%d_wd=%.1e_kpivot=0.3.txt'%(root_out, model, hidden, wd)
f_model  = '%s/models/%s_%d_wd=%.1e_kpivot=0.3.pt'%(root_out, model, hidden, wd)
######################################################################################

# use GPUs if available
GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s  ||  Training on %s'%(GPU, device))
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
train_loader = data.create_dataset_maps('train', seed, f_in, f_out, batch_size, 
                                        verbose=True)
valid_loader = data.create_dataset_maps('valid', seed, f_in, f_out, batch_size, 
                                        verbose=True)
test_loader  = data.create_dataset_maps('test',  seed, f_in, f_out, batch_size, 
                                        verbose=True)

# define the model and get total number of parameters
if   model=='model_a':  model = architecture.model_a(hidden).to(device)
elif model=='model_b':  model = architecture.model_b(hidden).to(device)
elif model=='model_c':  model = architecture.model_c(hidden).to(device)
elif model=='model_d':  model = architecture.model_d(hidden).to(device)
elif model=='model_e':  model = architecture.model_e(hidden).to(device)
elif model=='model_f':  model = architecture.model_f(hidden).to(device)
else:                   raise Exception('model not supported')
network_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model: %d'%network_total_params)

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)

# get validation loss
if os.path.exists(f_model):  model.load_state_dict(torch.load(f_model))
print('Computing initial validation loss')
model.eval()
min_valid_loss, points = 0.0, 0
for x, y in valid_loader:
    with torch.no_grad():
        x    = x.to(device=device)
        y    = y.to(device=device)
        y_NN = model(x)
        min_valid_loss += (criterion(y_NN, y).item())*x.shape[0]
        points += x.shape[0]
min_valid_loss /= points
print('Initial valid loss = %.3e'%min_valid_loss)

# see if results for this model are available
if os.path.exists(f_loss):  
    dumb = np.loadtxt(f_loss, unpack=False)
    offset = int(dumb[:,0][-1]+1)
else:   offset = 0


# do a loop over all the epochs
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
    if valid_loss<min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), f_model)
        print('Epoch %d: %.3e %.3e %.3e (saving)'%(epoch, train_loss, valid_loss, 
                                                   test_loss))
    else:
        print('Epoch %d: %.3e %.3e %.3e'%(epoch, train_loss, valid_loss, test_loss))

    # save losses
    f = open(f_loss, 'a')  
    f.write('%d %.3e %.3e\n'%(epoch, train_loss, valid_loss))
    f.close()


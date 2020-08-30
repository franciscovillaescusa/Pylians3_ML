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
f_in  = '%s/Gaussian_maps_test_A=1.00.npy'%root 
f_out = '%s/A_values_test_A=1.00.npy'%root
seed  = 20 #to split images between training, validation and testing sets

# architecture parameters
hidden     = 16
dr         = 0.0
batch_size = 1
model      = 'model_f'

# output files
root_out  = '/mnt/ceph/users/camels/Results/Gaussian_fields'
#wd        = 2e-4
#f_model   = '%s/models/%s_%d_wd=%.1e.pt'%(root_out, model, hidden, wd)
#f_map_out = '%s/saliency_maps/saliency_map_0.npy'%root_out

wd        = 4e-4
f_model   = '%s/models/%s_%d_wd=%.1e_kpivot=0.3_discon_0.3variation.pt'%(root_out, model, hidden, wd)
f_map_out = '%s/saliency_maps/saliency_map_0_kpivot=0.3_discon_0.3variation.npy'%root_out
######################################################################################

# use GPUs if available
GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s  ||  Training on %s'%(GPU, device))
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
test_loader  = data.create_dataset_maps('all',  seed, f_in, f_out, batch_size, 
                                        shuffle=False, verbose=True)

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

# load best model
if os.path.exists(f_model):  model.load_state_dict(torch.load(f_model))

# testing
model.eval()
for x,y in test_loader:
    #with torch.no_grad():
    x = x.to(device)
    y = y.to(device)
    #break
    
    x.requires_grad_()
    y_NN = model(x)
    y_NN.backward()
    loss = criterion(y_NN, y)
    
    saliency = (x.grad[0,0]).cpu().detach().numpy()
    x = (x[0,0]).cpu().detach().numpy()
    
    print(y.cpu().detach().numpy(), y_NN.cpu().detach().numpy())
    print('loss = %.3e'%loss)
    print(x.shape)
    print(saliency.shape)
    
    #np.save('image3.npy',x)
    np.save(f_map_out, saliency)
    break

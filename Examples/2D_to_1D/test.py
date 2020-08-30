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
#f_in  = '%s/Gaussian_maps.npy'%root 
#f_out = '%s/A_values.npy'%root
#f_in  = '%s/Gaussian_maps_test_A=1.00.npy'%root 
#f_out = '%s/A_values_test_A=1.00.npy'%root
f_in  = '%s/Gaussian_maps_kpivot=0.3_discon.npy'%root 
f_out = '%s/A_values_kpivot=0.3_discon.npy'%root
seed  = 20 #to split images between training, validation and testing sets

# architecture parameters
hidden     = 16
dr         = 0.0
wd         = 4e-4 #2e-4
batch_size = 128
model      = 'model_f'

# output files
root_out  = '/mnt/ceph/users/camels/Results/Gaussian_fields'
#f_model   = '%s/models/%s_%d_wd=%.1e.pt'%(root_out, model, hidden, wd)
#f_results = '%s/results/%s_%d_wd=%.1e.txt'%(root_out, model, hidden, wd)
#f_model   = '%s/models/%s_%d_wd=%.1e.pt'%(root_out, model, hidden, wd)
#f_results = '%s/results/%s_%d_wd=%.1e_A=1.00.txt'%(root_out, model, hidden, wd)
f_model  = '%s/models/%s_%d_wd=%.1e_kpivot=0.3_discon_0.3variation.pt'%(root_out, model, hidden, wd)
f_results  = '%s/results/%s_%d_wd=%.1e_kpivot=0.3_discon_0.3variation.txt'%(root_out, model, hidden, wd)
######################################################################################

# use GPUs if available
GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s  ||  Training on %s'%(GPU, device))
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data
#test_loader = data.create_dataset_maps('all', seed, f_in, f_out, batch_size, 
#                                       verbose=True)
test_loader = data.create_dataset_maps('test', seed, f_in, f_out, batch_size, 
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

# get the number of points in the test set
size = 0
for x,y in test_loader:  size += x.shape[0]

# load best-models, if they exists
if os.path.exists(f_model):  model.load_state_dict(torch.load(f_model, 
                                            map_location=torch.device('cpu')))

results = np.zeros((size,2), dtype=np.float32)

# test
test_loss, num_points = 0.0, 0
model.eval()
for x,y in test_loader:
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        y_NN = model(x)
        loss = criterion(y_NN, y)
        test_loss += (loss.cpu().item())*x.shape[0]

        results[num_points:num_points+x.shape[0],0] = y[:,0].cpu().numpy()
        results[num_points:num_points+x.shape[0],1] = y_NN[:,0].cpu().numpy()

        num_points += x.shape[0]
test_loss = test_loss/num_points

print('test loss = %.3e'%test_loss)
np.savetxt(f_results, results)


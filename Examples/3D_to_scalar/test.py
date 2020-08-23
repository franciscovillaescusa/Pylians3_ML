import numpy as np
import sys,os
import torch
import torch.nn as nn
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
hidden     = 16      #number of hidden layers; network parameter
batch_size = 128     #batch size

# output files
f_model   = 'models/model_b_6x_%s_dr=0.0_wd=2e-2.pt'%hidden
f_results = 'results/model_b_6x_%s_dr=0.0_wd=2e-2.txt'%hidden
######################################################################################

# use GPUs if available
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s     ||    Training on %s\n'%(GPU,device))
cudnn.benchmark = True      #May train faster but cost more memory

# define loss function
criterion = nn.MSELoss()

# get the data and its size
test_loader  = data.create_dataset_cubes('test',  seed, f_in, f_out, batch_size, 
                                         verbose=True)
size = len(test_loader)

# define the array with the results
pred = np.zeros((size, y.shape[1]), dtype=np.float32)
true = np.zeros((size, y.shape[1]), dtype=np.float32)

# define the model
#model = architecture.model_a(hidden).to(device)
model = architecture.model_b(hidden).to(device)

# load best-models, if they exists
if os.path.exists(f_model):  model.load_state_dict(torch.load(f_model,
                                            map_location=torch.device(device)))
else:                        raise Exception('File not found!!')

# get the test loss
model.eval()
test_loss, num_points, offset = 0.0, 0, 0
for x,y in test_loader:
    with torch.no_grad():
        x    = x.to(device=device)
        y    = y.to(device=device)
        y_NN = model(x)
        test_loss += (criterion(y_NN, y).item())*x.shape[0]
        num_points += x.shape[0]

        #fill arrays with results
        length = x.shape[0]
        pred[offset:offset+length] = y_NN.cpu().numpy()
        true[offset:offset+length] = y.cpu().numpy()
        offset += length

test_loss /= num_points
print('test loss = %.3e\n'%test_loss)


# denormalize values
A = np.load(f_out)
mean, std = np.mean(A, dtype=np.float64), np.std(A, dtype=np.float64)
pred = pred*std + mean
true = true*std + mean
results = np.zeros((pred.shape[0], true.shape[1]+pred.shape[1]), dtype=np.float64)
results[:,:true.shape[1]] = true
results[:,true.shape[1]:] = pred

# save results to file
np.savetxt(f_results, results)

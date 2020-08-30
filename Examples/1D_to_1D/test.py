import numpy as np
import torch
import sys,os,h5py
sys.path.append('../')
import data as data
import architecture

#################################### INPUT ##########################################
# data parameters
fin  = '12features.hdf5'
seed = 5

# architecture parameters
h1 = 100
h2 = 400
h3 = 100
dropout_rate = 0.0

# training parameters
batch_size = 256

# name of output files
name   = '1hd_100_0.0_0.0'
fout   = 'results/%s.txt'%name
fmodel = 'models/%s.pt'%name
#####################################################################################

# get GPU if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print("CUDA Not Available")
    device = torch.device('cpu')

# get the test dataset
test_loader  = data.create_dataset('test', seed, fin, batch_size)

# get the number of elements in the test set
size = 0
for x, y in test_loader:
    size += x.shape[0]

# define the array with the results
pred = np.zeros((size,1), dtype=np.float32)
true = np.zeros((size,1), dtype=np.float32)

# get the parameters of the trained model
model = architecture.model_1hl(12, h1, 1, dropout_rate)
#model = architecture.model_2hl(12, h1, h2, 1, dropout_rate)
#model = architecture.model_3hl(12, h1, h2, h3, 1, dropout_rate)
model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
model.to(device=device)

# loop over the different batches and get the prediction
offset = 0
model.eval()
for x, y in test_loader:
    with torch.no_grad():
        x    = x.to(device)
        y    = y.to(device)
        y_NN = model(x)
        length = x.shape[0]
        pred[offset:offset+length] = y_NN.cpu().numpy()
        true[offset:offset+length] = y.cpu().numpy()
        offset += length

# get mean and std of M_HI
f = h5py.File(fin, 'r')
M_HI = f['M_HI'][:]
f.close()
M_HI = np.log10(1.0 + M_HI)
mean, std = np.mean(M_HI), np.std(M_HI)
print(std)

# compute the rmse; de-normalize
error_norm = ((pred - true))**2
pred  = pred*std + mean
true  = true*std + mean
error = (pred - true)**2

print('Error^2 norm      = %.3e'%np.mean(error_norm))
print('Error             = %.3e'%np.sqrt(np.mean(error)))

# save results to file
#results = np.zeros((size,10))
#results[:,0:5]  = true
#results[:,5:10] = pred
#np.savetxt(fout, results)


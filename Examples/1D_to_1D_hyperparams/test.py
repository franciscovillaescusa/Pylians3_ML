import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna

################################### INPUT ############################################
# data parameters
f_Pk      = '/mnt/ceph/users/camels/Software/LFI_GNN/data_preprocessing/Pk_galaxies_SIMBA_LH_33_kmax=20.0.npy'
f_Pk_norm = '/mnt/ceph/users/camels/Software/LFI_GNN/data_preprocessing/Pk_galaxies_IllustrisTNG_LH_33_kmax=20.0.npy'
f_params  = '/mnt/ceph/users/camels/Software/SIMBA/latin_hypercube_params.txt' 
seed      = 1
mode      = 'all'

# architecture parameters
input_size         = 79
output_size        = 2
max_layers         = 5
max_neurons_layers = 1000

# training parameters
batch_size = 32
epochs     = 1000

# optuna parameters
study_name = 'Pk_2_params'
storage    = 'sqlite:///TPE.db'
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# load the optuna study
study = optuna.load_study(study_name=study_name, storage=storage)

# get the scores of the study trials
values = np.zeros(len(study.trials))
completed = 0
for i,t in enumerate(study.trials):
    values[i] = t.value
    if t.value is not None:  completed += 1

# get the info of the best trial
indexes = np.argsort(values)
for i in range(1):
    trial = study.trials[indexes[i]]
    print("\nTrial number {}".format(trial.number))
    print("Value: %.5e"%trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    n_layers = trial.params['n_layers']
    lr       = trial.params['lr']
    wd       = trial.params['wd']
    hidden   = np.zeros(n_layers, dtype=np.int32)
    dr       = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        hidden[i] = trial.params['n_units_l%d'%i]
        dr[i]     = trial.params['dropout_l%d'%i]
    fmodel = 'models/model_%d.pt'%trial.number

# generate the architecture
model = architecture.dynamic_model2(input_size, output_size, n_layers, hidden, dr)
model.to(device)    

# load best-model, if it exists
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
else:
    raise Exception('model doesnt exists!!!')

# define loss function
criterion = nn.MSELoss() 

# get the data
test_loader = data.create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, 
                                  batch_size, shuffle=False)
points = 0
for x,y in test_loader:  points += x.shape[0]

# define the arrays containing the true and predicted value of the parameters
params_true = np.zeros((points,output_size), dtype=np.float32)
params_pred = np.zeros((points,output_size), dtype=np.float32)

# test the model
test_loss, points = 0.0, 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_NN = model(x)
        test_loss += (criterion(y_NN, y).item())*x.shape[0]
        params_true[points:points+x.shape[0]] = y.cpu().numpy()
        params_pred[points:points+x.shape[0]] = y_NN.cpu().numpy()
        points     += x.shape[0]
test_loss /= points

print(test_loss)

# save results to file
results = np.zeros((params_true.shape[0],2*params_true.shape[1]), dtype=np.float32)
results[:,:params_true.shape[1]] = params_true
results[:,params_true.shape[1]:] = params_pred

np.savetxt('borrar.txt', results)

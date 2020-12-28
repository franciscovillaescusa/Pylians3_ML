import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna


class Objective(object):
    def __init__(self, input_size, output_size, max_layers, max_neurons_layers, device,
                 epochs, seed, realizations, root_in, bins_SFRH, sim, batch_size, 
                 root_out):

        self.input_size         = input_size
        self.output_size        = output_size
        self.max_layers         = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device             = device
        self.epochs             = epochs
        self.seed               = seed
        self.realizations       = realizations
        self.root_in            = root_in
        self.bins_SFRH          = bins_SFRH
        self.sim                = sim
        self.batch_size         = batch_size
        self.root_out           = root_out

    def __call__(self, trial):

        # name of the files that will contain the losses and model weights
        fout   = 'losses/loss_%d.txt'%(trial.number)
        fmodel = 'models/model_%d.pt'%(trial.number)

        # generate the architecture
        model = architecture.dynamic_model(trial, self.input_size, self.output_size, 
                            self.max_layers, self.max_neurons_layers).to(self.device)

        # get the weight decay and learning rate values
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                     weight_decay=wd)

        # define loss function
        criterion = nn.MSELoss() 

        # get the data
        train_loader = data.create_dataset('train', self.seed, self.realizations, 
                                           self.root_in, self.bins_SFRH, self.sim, 
                                           self.batch_size, self.root_out)
        valid_loader = data.create_dataset('valid', self.seed, self.realizations, 
                                           self.root_in, self.bins_SFRH, self.sim, 
                                           self.batch_size, self.root_out)

        # train/validate model
        min_valid = 1e40
        for epoch in range(self.epochs):

            # training
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_NN = model(x)
                loss = criterion(y_NN, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validation
            valid_loss, points = 0.0, 0
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_NN = model(x)
                    valid_loss += (criterion(y_NN, y).item())*x.shape[0]
                    points     += x.shape[0]
            valid_loss /= points

            if valid_loss<min_valid:  
                min_valid = valid_loss
                torch.save(model.state_dict(), fmodel)
            f = open(fout, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, valid_loss, min_valid))
            f.close()

            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            #trial.report(min_valid, epoch)
            #if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        return min_valid

##################################### INPUT ##########################################
# data parameters
root_in      = '/mnt/ceph/users/camels'
root_out     = '/mnt/ceph/users/camels/Results/neural_nets/params_2_SFRH/SIMBA'
sim          = 'SIMBA'
seed         = 1
realizations = 1000
bins_SFRH    = 100

# architecture parameters
input_size         = 6
output_size        = bins_SFRH
max_layers         = 5
max_neurons_layers = 1000

# training parameters
batch_size = 256
epochs     = 600

# optuna parameters
study_name = 'params_2_SFRH_TPE'
n_trials   = 1000 #set to None for infinite
storage    = 'sqlite:///TPE.db'
n_jobs     = 28
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define the optuna study and optimize it
objective = Objective(input_size, output_size, max_layers, max_neurons_layers, device,
            epochs, seed, realizations, root_in, bins_SFRH, sim, batch_size, root_out)
#sampler = optuna.samplers.RandomSampler()
sampler = optuna.samplers.TPESampler(n_startup_trials=10)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage)
study.optimize(objective, n_trials, n_jobs=n_jobs)


# get the number of pruned and complete trials
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# print some verbose
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials:   ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

# parameters of the best trial
trial = study.best_trial
print("Best trial: number {}".format(trial.number))
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# This script load a study and output some information about it
import numpy as np
import optuna

################################## INPUT ########################################
# study parameters
study_name = 'params_2_SFRH_TPE'
storage    = 'sqlite:///TPE.db'

# whether print information of a particular trial
individual_trials = None #if None, nothing is printed
#################################################################################

# load the study
study = optuna.load_study(study_name=study_name, storage=storage)

# get the number of trials
trials = len(study.trials)
print('Found %d trials'%trials)

# get the scores of the trials and print the top-10
values = np.zeros(trials)
for i,t in enumerate(study.trials):
    values[i] = t.value

indexes = np.argsort(values)
print('\nBest trials:')
for i in range(10):
    print('study %04d ----> score: %.5e'%(indexes[i], values[indexes[i]]))

# get the info of the best trial
trial = study.best_trial
print("\nBest trial:  number {}".format(trial.number))
print("  Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

if individual_trials is not None:
    for trial_number in individual_trials:
        trial = study.trials[trial_number]
        print(" \nTrial number {}".format(trial_number))
        print("  Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

"""
# read some parameters from the trials and save results to a .txt file
lr   = np.zeros(10000, dtype=np.float64)
wd   = np.zeros(10000, dtype=np.float64)
nl   = np.zeros(10000, dtype=np.float64)
loss = np.zeros(10000, dtype=np.float64)
for i,t in enumerate(study.trials):
    if i>=10000:  break
    lr[i]   = t.params['lr']
    wd[i]   = t.params['wd']
    nl[i]   = t.params['n_layers']
    loss[i] = t.value
np.savetxt('borrar.txt',np.transpose([np.arange(10000),nl,lr,wd,loss]))
"""
"""
# get the importances of the different hyperparameters
importances = optuna.importance.get_param_importances(study)
print('\nImportances:')
for key in importances:
    print('{} {}'.format(key, importances[key]))
"""




import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, f_Pk, f_Pk_norm, f_params):

        # read data, scale it, and normalize it
        Pk = np.log10(np.load(f_Pk))
        if f_Pk_norm is None:
            mean, std = np.mean(Pk, axis=0), np.std(Pk, axis=0)
        else:
            Pk_norm = np.log10(np.load(f_Pk))
            mean, std = np.mean(Pk_norm, axis=0), np.std(Pk_norm, axis=0)
        Pk = (Pk - mean)/std

        # read the value of the cosmological & astrophysical parameters; normalize them
        params  = np.loadtxt(f_params)
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        sims = Pk.shape[0]
        if   mode=='train':  size, offset = int(sims*0.70), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.15), int(sims*0.70)
        elif mode=='test':   size, offset = int(sims*0.15), int(sims*0.85)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')

        # randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(sims) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # select the data in the considered mode
        Pk     = Pk[indexes]
        params = params[indexes,:2]

        # define size, input and output matrices
        self.size   = size
        self.input  = torch.tensor(Pk,     dtype=torch.float)
        self.output = torch.tensor(params, dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
# mode ---------------> 'train', 'valid', 'test' or 'all'
# seed ---------------> random seed to split data among training, validation and testing
# f_Pk ---------------> file containing the power spectra
# f_Pk_norm ----------> file containing the power spectra to normalize data
# f_params -----------> files with the value of the cosmological + astrophysical params
# batch_size ---------> batch size
# shuffle ------------> whether to shuffle the data or not
def create_dataset(mode, seed, f_Pk, f_Pk_norm, f_params, batch_size, shuffle):
    data_set = make_dataset(mode, seed, f_Pk, f_Pk_norm, f_params)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)

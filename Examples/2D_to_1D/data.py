import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

###################################################################################
# This class creates the dataset for Pk
class make_dataset_Pk():

    def __init__(self, mode, seed, f_in, f_out, verbose=False):

        # read the data
        x = np.load(f_in) #[number of Pk, Pk]
        y = np.load(f_out)

        # normalize Pk
        x = np.log10(x)
        minimum, maximum = np.min(x), np.max(x)
        #data = 2*(data - minimum)/(maximum-minimum) - 1.0
        if verbose:  print('%.3f < Pk < %.3f'%(np.min(x), np.max(x))) 

        # get the size and offset depending on the type of dataset
        unique_points = x.shape[0]
        if   mode=='train':  
            size, offset = int(unique_points*0.70), int(unique_points*0.00)
        elif mode=='valid':  
            size, offset = int(unique_points*0.15), int(unique_points*0.70)
        elif mode=='test':   
            size, offset = int(unique_points*0.15), int(unique_points*0.85)
        elif mode=='all':
            size, offset = int(unique_points*1.00), int(unique_points*0.00)
        else:    raise Exception('Wrong name!')

        # randomly shuffle the maps. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(unique_points) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # keep only the data with the corresponding indexes
        x = x[indexes]
        y = y[indexes]
            
        if verbose:
            print('A total of %d Pk used'%len(indexes))
            print('%.3f < Pk < %.3f'%(np.min(x), np.max(x)))
            print('%.3f < A  < %.3f\n'%(np.min(y), np.max(y)))

        self.size = x.shape[0]
        self.x    = torch.tensor(x, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.float32)
        #self.numbers = torch.tensor(numbers_all, dtype=torch.int32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
###################################################################################

###################################################################################
# This class creates the dataset for Pk
class make_dataset_maps():

    def __init__(self, mode, seed, f_in, f_out, verbose=False):

        # read the data
        x = np.load(f_in)  #[number of maps, height, width]
        y = np.load(f_out) #[A]

        # we compute mean and std from the whole data set, not individual sets
        x_norm = np.load('/mnt/ceph/users/camels/Results/Gaussian_fields/data/Gaussian_maps.npy')
        mean, std = np.mean(x_norm, dtype=np.float64), np.std(x_norm, dtype=np.float64)

        # normalize maps; take into account that mean of each map should be 0
        x /= std
        if verbose:  print('%.3f < delta < %.3f'%(np.min(x), np.max(x))) 

        # get the size and offset depending on the type of dataset
        unique_maps = x.shape[0]
        if   mode=='train':  
            size, offset = int(unique_maps*0.70), int(unique_maps*0.00)
        elif mode=='valid':  
            size, offset = int(unique_maps*0.15), int(unique_maps*0.70)
        elif mode=='test':   
            size, offset = int(unique_maps*0.15), int(unique_maps*0.85)
        elif mode=='all':
            size, offset = int(unique_maps*1.00), int(unique_maps*0.00)
        else:    raise Exception('Wrong name!')

        # randomly shuffle the maps. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(unique_maps) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # keep only the data with the corresponding indexes
        x = torch.tensor(x[indexes], dtype=torch.float32)
        y = torch.tensor(y[indexes], dtype=torch.float32)
        x = torch.unsqueeze(x, 1) #[number_of_maps, channels, height, width]
        y = torch.unsqueeze(y, 1) #[number_of_A,    1]
            
        if verbose:
            print('A total of %d maps used'%len(indexes))
            print('%.3f < delta < %.3f'%(torch.min(x), torch.max(x)))
            print('%.3f <   A   < %.3f\n'%(torch.min(y), torch.max(y)))

        self.size = size
        self.x    = x
        self.y    = y

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
###################################################################################

# This routine creates a dataset for Pk
def create_dataset_Pk(mode, seed, f_in, f_out, batch_size, verbose=False):
    data_set    = make_dataset_Pk(mode, seed, f_in, f_out, verbose)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader

# This routine creates a dataset for the maps
def create_dataset_maps(mode, seed, f_in, f_out, batch_size,shuffle=True,verbose=False):
    data_set    = make_dataset_maps(mode, seed, f_in, f_out, verbose)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader



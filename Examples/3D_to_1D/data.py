import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time
import data_operations as DO

###################################################################################
# This class creates the dataset for the 3D Gaussian density fields
# mode ----------> 'train', 'valid', 'test' or 'all'
# seed ----------> random seed; used to randomly split among train/valid/test
# f_in ----------> file containing the input data
# f_out ---------> file containing the output data
# verbose -------> whether print some information on the progress
class make_dataset_cubes():
    def __init__(self, mode, seed, f_in, f_out, verbose=False):

        # read the original data
        x_orig = np.load(f_in)  #[number of cubes, x, y, z]
        y_orig = np.load(f_out) #[A]

        # define the arrays containing the augmented data
        cubes = x_orig.shape[0]
        x = np.zeros((cubes*6, x_orig.shape[1], x_orig.shape[2], x_orig.shape[3]), 
                     dtype=np.float32)
        y = np.zeros(cubes*6, dtype=np.float32)
        
        # do all rotations and flipings
        num = 0
        for i in range(cubes):
            for j in range(6):
                x[num] = DO.rotation_flip_3D(x_orig[i], j%24, j//24)
                y[num] = y_orig[i]
                num += 1
        if verbose:  print('A total of %d cubes generated'%num)
        del x_orig, y_orig

        # normalize cubes; since mean is basically 0, just use std
        x /= np.std(x, dtype=np.float64)
        y = (y - np.mean(y, dtype=np.float64))/np.std(y, dtype=np.float64)
        if verbose:  
            print('< delta normalized > = %.3e'%np.mean(x, dtype=np.float64))
            print('%.3f < delta normalized < %.3f'%(np.min(x), np.max(x))) 
            print('< A normalized > = %.3e'%np.mean(y, dtype=np.float64))
            print('%.3f <   A normalized   < %.3f'%(np.min(y), np.max(y)))
            print('#################################')

        # get the size and offset depending on the type of dataset
        unique_cubes = x.shape[0]
        if   mode=='train':  
            size, offset = int(unique_cubes*0.70), int(unique_cubes*0.00)
        elif mode=='valid':  
            size, offset = int(unique_cubes*0.15), int(unique_cubes*0.70)
        elif mode=='test':   
            size, offset = int(unique_cubes*0.15), int(unique_cubes*0.85)
        elif mode=='all':
            size, offset = int(unique_cubes*1.00), int(unique_cubes*0.00)
        else:    raise Exception('Wrong name!')

        # randomly shuffle the cubes. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(unique_cubes) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # keep only the data with the corresponding indexes
        x = torch.tensor(x[indexes], dtype=torch.float32)
        y = torch.tensor(y[indexes], dtype=torch.float32)
        x = torch.unsqueeze(x, 1) #[number_of_cubes, channels, x, y, z]
        y = torch.unsqueeze(y, 1) #[number_of_A,    1]
            
        if verbose:
            print('A total of %d cubes used for %s'%(len(indexes),mode))
            print('%.3f < delta %s < %.3f'%(torch.min(x), mode, torch.max(x)))
            print('%.3f <   A %s   < %.3f\n'%(torch.min(y), mode, torch.max(y)))

        self.size = size
        self.x    = x
        self.y    = y

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
###################################################################################

# This routine creates a dataset for the 3D cubes
def create_dataset_cubes(mode, seed, f_in, f_out, batch_size, verbose=False):
    data_set    = make_dataset_cubes(mode, seed, f_in, f_out, verbose)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader



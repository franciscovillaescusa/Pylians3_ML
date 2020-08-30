import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time, h5py

def normalize_data(data, labels):

    # some information
    """
    print(data.shape)
    print('%.3e <   Zg   < %.3e   %.3e'%(np.min(data[:,0]),  np.max(data[:,0]),  np.mean(data[:,0])))
    print('%.3e <   Zs   < %.3e   %.3e'%(np.min(data[:,1]),  np.max(data[:,1]),  np.mean(data[:,1])))
    print('%.3e <   Ns   < %.3e   %.3e'%(np.min(data[:,2]),  np.max(data[:,2]),  np.mean(data[:,2])))
    print('%.3e <  SFR   < %.3e   %.3e'%(np.min(data[:,3]),  np.max(data[:,3]),  np.mean(data[:,3])))
    print('%.3e <   V    < %.3e   %.3e'%(np.min(data[:,4]),  np.max(data[:,4]),  np.mean(data[:,4])))
    print('%.3e <  M_BH  < %.3e   %.3e'%(np.min(data[:,5]),  np.max(data[:,5]),  np.mean(data[:,5])))
    print('%.3e <   M    < %.3e   %.3e'%(np.min(data[:,6]),  np.max(data[:,6]),  np.mean(data[:,6])))
    print('%.3e < rho_1  < %.3e   %.3e'%(np.min(data[:,7]),  np.max(data[:,7]),  np.mean(data[:,7])))
    print('%.3e < rho_2  < %.3e   %.3e'%(np.min(data[:,8]),  np.max(data[:,8]),  np.mean(data[:,8])))
    print('%.3e < rho_5  < %.3e   %.3e'%(np.min(data[:,9]),  np.max(data[:,9]),  np.mean(data[:,9])))
    print('%.3e < rho_10 < %.3e   %.3e'%(np.min(data[:,10]), np.max(data[:,10]), np.mean(data[:,10])))
    print('%.3e < rho_20 < %.3e   %.3e'%(np.min(data[:,11]), np.max(data[:,11]), np.mean(data[:,11])))
    print('%.3e <  M_HI  < %.3e   %.3e'%(np.min(labels),     np.max(labels),     np.mean(labels)))
    """

    # normalize input
    data[:,0]  = (data[:,0] - np.mean(data[:,0]))/np.std(data[:,0])  #Zg
    data[:,1]  = (data[:,1] - np.mean(data[:,1]))/np.std(data[:,1])  #Zs
    data[:,2]  = (data[:,2] - np.mean(data[:,2]))/np.std(data[:,2])  #Ns
    data[:,3]  = (data[:,3] - np.mean(data[:,3]))/np.std(data[:,3])  #SFR
    data[:,4]  = (data[:,4] - np.mean(data[:,4]))/np.std(data[:,4])  #V
    array      = np.log10(1.0+data[:,5])
    data[:,5]  = (array - np.mean(array))/np.std(array)
    array      = np.log10(1.0+data[:,6])
    data[:,6]  = (array - np.mean(array))/np.std(array)
    data[:,7]  = (data[:,7]  - np.mean(data[:,7]))/np.std(data[:,7])    #rho1
    data[:,8]  = (data[:,8]  - np.mean(data[:,8]))/np.std(data[:,8])    #rho2
    data[:,9]  = (data[:,9]  - np.mean(data[:,9]))/np.std(data[:,9])    #rho5
    data[:,10] = (data[:,10] - np.mean(data[:,10]))/np.std(data[:,10])  #rho10
    data[:,11] = (data[:,11] - np.mean(data[:,11]))/np.std(data[:,11])  #rho20

    # normalize labels
    array = np.log10(1.0 + labels)
    labels = (array - np.mean(array))/np.std(array)

    return data, labels

# read data and get training, validation or testing sets
# fin ---------> file with the data
# seed --------> random seed used to split among different datasets
# mode --------> 'train', 'valid', 'test' or 'all'
# normalize ---> whether to normalize the data or not
def read_data(fin, seed, mode, normalize):

    # read data
    f     = h5py.File(fin, 'r')
    Zg    = f['Gas Metallicity'][:];  #Zg = np.log10(1.0+Zg);  Zg = (Zg - np.mean(Zg))/np.std(Zg)
    Zs    = f['Star Metallicity'][:]; #Zs = np.log10(1.0+Zs);  Zs = (Zs - np.mean(Zs))/np.std(Zs)
    Ns    = f['N_subhalos'][:];       #Ns = np.log10(1.0+Ns);  Ns = (Ns - np.mean(Ns))/np.std(Ns)
    SFR   = f['SFR'][:];              #SFR = np.log10(1.0+SFR);  SFR = (SFR - np.mean(SFR))/np.std(SFR)
    V     = f['Velocity'][:];         #V = np.log10(1.0+V);  V = (V - np.mean(V))/np.std(V)
    rho1  = f['rho_1'][:];            #rho1 = np.log10(1.0+rho1);  rho1 = (rho1 - np.mean(rho1))/np.std(rho1)
    rho2  = f['rho_2'][:];            #rho2 = np.log10(1.0+rho2);  rho2 = (rho2 - np.mean(rho2))/np.std(rho2)
    rho5  = f['rho_5'][:];            #rho5 = np.log10(1.0+rho5);  rho5 = (rho5 - np.mean(rho5))/np.std(rho5)
    rho10 = f['rho_10'][:];           #rho10 = np.log10(1.0+rho10);  rho10 = (rho10 - np.mean(rho10))/np.std(rho10)
    rho20 = f['rho_20'][:];           #rho20 = np.log10(1.0+rho20);  rho20 = (rho20 - np.mean(rho20))/np.std(rho20)
    M_BH  = f['M_BH'][:];             #M_BH = np.log10(1.0+M_BH);  M_BH = (M_BH - np.mean(M_BH))/np.std(M_BH)
    M     = f['M_halo'][:];           #M = np.log10(1.0+M);  M = (M - np.mean(M))/np.std(M)
    M_HI  = f['M_HI'][:];             #M_HI = np.log10(1.0+M_HI);  M_HI = (M_HI - np.mean(M_HI))/np.std(M_HI)
    f.close()

    # get data, labels and number of elements
    data     = np.vstack([Zg, Zs, Ns, SFR, V, M_BH, M, rho1, rho2, rho5, rho10, rho20]).T
    labels   = M_HI.reshape((M_HI.shape[0],1))
    elements = data.shape[0]

    # normalize data
    if normalize:  data, labels = normalize_data(data, labels)

    # get the size and offset depending on the type of dataset
    if   mode=='train':   size, offset = int(elements*0.70), int(elements*0.00)
    elif mode=='valid':   size, offset = int(elements*0.15), int(elements*0.70)
    elif mode=='test':    size, offset = int(elements*0.15), int(elements*0.85)
    elif mode=='all':     size, offset = int(elements*1.00), int(elements*0.00)
    else:                 raise Exception('Wrong name!')

    # randomly shuffle the cubes. Instead of 0 1 2 3...999 have a 
    # random permutation. E.g. 5 9 0 29...342
    np.random.seed(seed)
    indexes = np.arange(elements) 
    np.random.shuffle(indexes)
    indexes = indexes[offset:offset+size] #select indexes of mode

    return data[indexes], labels[indexes]


# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, fin):

        # get data
        inp, out = read_data(fin, seed, mode, normalize=True)

        # get the corresponding bottlenecks and parameters
        self.size   = inp.shape[0]
        self.input  = torch.tensor(inp, dtype=torch.float32)
        self.output = torch.tensor(out, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
def create_dataset(mode, seed, fin, batch_size):
    data_set = make_dataset(mode, seed, fin)
    dataset_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return dataset_loader

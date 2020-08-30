import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data as data
import architecture as architecture

##################################### INPUT ##########################################
# data parameters
fin  = '12features.hdf5'
seed = 5

# architecture parameters
h1 = 2000
h2 = 2000
h3 = 2000
h4 = 1000
dropout_rate = 0.3

# training parameters
batch_size = 256
lr         = 1e-4
epochs     = 100000
wd         = 1e-5

# name of output files
name   = '1hd_2000_0.3_1e-5'
fout   = 'losses/%s.txt'%name
fmodel = 'models/%s.pt'%name
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define loss function
criterion = nn.MSELoss() 

# get train and validation sets
print('preparing dataset...')
train_loader = data.create_dataset('train', seed, fin, batch_size)
valid_loader = data.create_dataset('valid', seed, fin, batch_size)
test_loader  = data.create_dataset('test',  seed, fin, batch_size)

# define architecture
model = architecture.model_1hl(12, h1, 1, dropout_rate)
#model = architecture.model_2hl(12, h1, h2, 1, dropout_rate)
#model = architecture.model_3hl(12, h1, h2, h3, 1, dropout_rate)
#model = architecture.model_4hl(12, h1, h2, h3, h4, 1, dropout_rate)
model.to(device=device)
network_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model = %d'%network_total_params)

# define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                             weight_decay=wd)	

# load best-model, if it exists
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel))

# get validation loss
print('Computing initial validation loss')
model.eval()
min_valid_loss, points = 0.0, 0
for x, y in valid_loader:
    with torch.no_grad():
        x    = x.to(device=device)
        y    = y.to(device=device)
        y_NN = model(x)
        min_valid_loss += (criterion(y_NN, y).item())*x.shape[0]
        points += x.shape[0]
min_valid_loss /= points
print('Initial valid loss = %.3e'%min_valid_loss)

# see if results for this model are available
if os.path.exists(fout):  
    dumb = np.loadtxt(fout, unpack=False)
    offset = int(dumb[:,0][-1]+1)
else:   offset = 0

# do a loop over all epochs
start = time.time()
for epoch in range(offset, offset+epochs):
        
    # do training
    train_loss, points = 0.0, 0
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_NN = model(x)

        loss = criterion(y_NN, y)
        train_loss += (loss.item())*x.shape[0]
        points     += x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= points

    # do validation
    valid_loss, points = 0.0, 0
    model.eval()
    for x, y in valid_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            valid_loss += (criterion(y_NN, y).item())*x.shape[0]
            points     += x.shape[0]
    valid_loss /= points

    # do testing
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            test_loss += (criterion(y_NN, y).item())*x.shape[0]
            points    += x.shape[0]
    test_loss /= points

    # save model if it is better
    if valid_loss<min_valid_loss:
        torch.save(model.state_dict(), fmodel)
        min_valid_loss = valid_loss
        print('%03d %.3e %.3e %.3e (saving)'%(epoch, train_loss, valid_loss, test_loss))
    else:
        print('%03d %.3e %.3e %.3e'%(epoch, train_loss, valid_loss, test_loss))
    
    # save losses to file
    f = open(fout, 'a')
    f.write('%d %.5e %.5e %.5e\n'%(epoch, train_loss, valid_loss, test_loss))
    f.close()
    
stop = time.time()
print('Time take (m):', "{:.4f}".format((stop-start)/60.0))

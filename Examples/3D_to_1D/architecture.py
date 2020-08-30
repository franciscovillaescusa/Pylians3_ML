import torch 
import torch.nn as nn
import numpy as np
import sys, os, time


# This function implements periodic padding of the images
def pp(df, padding):
    
    #y
    #y
    #y
    #y
    #0 x x x x

    right = df[:,:,-padding:, :]
    left  = df[:,:,:padding, :]
    df = torch.cat((df,    left), dim=2)
    df = torch.cat((right, df),   dim=2)

    top    = df[:,:,:, -padding:]
    bottom = df[:,:,:, :padding]
    df = torch.cat((df,  bottom), dim=3)
    df = torch.cat((top, df),     dim=3)

    return df


######## 1 hidden layer ##########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_1hl(nn.Module):
    
    def __init__(self, inp, h1, out, dr):
        super(model_1hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.fc2(out)         
        return out
##################################

######## 2 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_2hl(nn.Module):
    
    def __init__(self, inp, h1, h2, out, dr):
        super(model_2hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.fc3(out)         
        return out
##################################

######## 3 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# h3 ----------> size of third  hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_3hl(nn.Module):
    
    def __init__(self, inp, h1, h2, h3, out, dr):
        super(model_3hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  h3)
        self.fc4 = nn.Linear(h3,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.fc4(out)         
        return out
##################################

######## 4 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# h3 ----------> size of third  hidden layer
# h4 ----------> size of fourth hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_4hl(nn.Module):
    
    def __init__(self, inp, h1, h2, h3, h4, out, dr):
        super(model_4hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  h3)
        self.fc4 = nn.Linear(h3,  h4)
        self.fc5 = nn.Linear(h4,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.dropout(self.LeakyReLU(self.fc4(out)))
        out = self.fc5(out)         
        return out
##################################



# architecture of model 1
class model1(nn.Module):
    
    def __init__(self, inp, out1, out2, out3, out4, out5, out6):
        super(model1, self).__init__()
        
        self.fc1 = nn.Linear(inp,  out1) 
        self.fc2 = nn.Linear(out1, out2)
        self.fc3 = nn.Linear(out2, out3)
        self.fc4 = nn.Linear(out3, out4)
        self.fc5 = nn.Linear(out4, out5)
        self.fc6 = nn.Linear(out5, out6)
	
        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.dropout(self.LeakyReLU(self.fc4(out)))
        out = self.dropout(self.LeakyReLU(self.fc5(out)))
        out = self.fc6(out)         
        return out


# architecture of model 2
class model2(nn.Module):
    
    def __init__(self, inp, out1, out2, out3):
        super(model2, self).__init__()
        
        self.fc1 = nn.Linear(inp,  out1) 
        self.fc2 = nn.Linear(out1, out2)
        self.fc3 = nn.Linear(out2, out3)
	
        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.fc3(out)         
        return out


# architecture of model 3
class model3(nn.Module):
    
    def __init__(self, inp, out1, out2):
        super(model3, self).__init__()
        
        self.fc1 = nn.Linear(inp,  out1) 
        self.fc2 = nn.Linear(out1, out2)
	
        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.fc2(out)         
        return out


#####################################################################################
#####################################################################################
class model_a(nn.Module):
    def __init__(self, hidden):
        super(model_a, self).__init__()
        
        # input: 1x32x32x32 ---------------> output: hiddenx16x16x16
        self.C1 = nn.Conv3d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm3d(hidden)
        # input: hiddenx16x16x16 ----------> output: 2*hiddenx8x8x8
        self.C2 = nn.Conv3d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm3d(2*hidden)
        # input: 2*hiddenx8x8x8 -----------> output: 4*hiddenx4x4x4
        self.C3 = nn.Conv3d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm3d(4*hidden)
        # input: 4*hiddenx4x4x4 -----------> output: 8*hiddenx2x2x2
        self.C4 = nn.Conv3d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm3d(8*hidden)
        # input: 8*hiddenx2x2x2 -----------> output: 16*hiddenx1x1x1
        self.C5 = nn.Conv3d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm3d(16*hidden)

        self.FC1  = nn.Linear(16*hidden, 250)  
        self.FC2  = nn.Linear(250,   100)    
        self.FC3  = nn.Linear(100,   1)    

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = x.view(image.shape[0],-1)
        x = self.LeakyReLU(self.FC1(x))
        x = self.LeakyReLU(self.FC2(x))
        x = self.FC3(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_b(nn.Module):
    def __init__(self, hidden):
        super(model_b, self).__init__()
        
        # input: 1x32x32x32 ---------------> output: hiddenx16x16x16
        self.C1 = nn.Conv3d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm3d(hidden)
        # input: hiddenx16x16x16 ----------> output: 2*hiddenx8x8x8
        self.C2 = nn.Conv3d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm3d(2*hidden)
        # input: 2*hiddenx8x8x8 -----------> output: 4*hiddenx4x4x4
        self.C3 = nn.Conv3d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm3d(4*hidden)
        # input: 4*hiddenx4x4x4 -----------> output: 8*hiddenx2x2x2
        self.C4 = nn.Conv3d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm3d(8*hidden)
        # input: 8*hiddenx2x2x2 -----------> output: 16*hiddenx1x1x1
        self.C5 = nn.Conv3d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm3d(16*hidden)

        self.FC1  = nn.Linear(16*hidden, 1)  

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################



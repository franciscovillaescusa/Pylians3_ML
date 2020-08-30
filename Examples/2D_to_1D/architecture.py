import torch 
import torch.nn as nn
import numpy as np
import sys, os, time


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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
                nn.init.constant_(m.bias, 0)
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
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 8*hiddenx8x8
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx4x4 ----------> output: 50x4x4
        self.C6 = nn.Conv2d(16*hidden, 50, kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.B6 = nn.BatchNorm2d(50)

        self.FC1  = nn.Linear(50*4*4, 400)  
        self.FC2  = nn.Linear(400,   100)    
        self.FC3  = nn.Linear(100,   1)    

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
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
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 8*hiddenx8x8
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx4x4 ----------> output: 50x4x4
        self.C6 = nn.Conv2d(16*hidden, 50, kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.B6 = nn.BatchNorm2d(50)

        self.FC1  = nn.Linear(50*4*4, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_c(nn.Module):
    def __init__(self, hidden):
        super(model_c, self).__init__()
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 8*hiddenx8x8
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx4x4 ----------> output: 32*hiddenx2x2
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)
        # input: 32*hiddenx4x4 ----------> output: 50x2x2
        self.C7 = nn.Conv2d(32*hidden, 50, kernel_size=3, stride=1, padding=1,
                            bias=True)
        self.B7 = nn.BatchNorm2d(50)

        self.FC1  = nn.Linear(50*2*2, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = self.LeakyReLU(self.B7(self.C7(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_d(nn.Module):
    def __init__(self, hidden):
        super(model_d, self).__init__()
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 10x8x8
        self.C4 = nn.Conv2d(4*hidden, 10, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(10*8*8)

        self.FC1  = nn.Linear(10*8*8, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.C4(x))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e(nn.Module):
    def __init__(self, hidden):
        super(model_e, self).__init__()
        
        # input: 1x128x128 ---------------> output: hiddenx43x43
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=6, stride=3, padding=2, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx43x43 ----------> output: 2*hiddenx22x22
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=2,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx22x22 --------> output: 4*hiddenx11x11
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx11x11 ----------> output: 8*hiddenx6x6
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=2,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx6x6 ------------> output: 16*hiddenx3x3
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)

        self.FC1  = nn.Linear(16*hidden*3*3, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
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

#####################################################################################
#####################################################################################
class model_f(nn.Module):
    def __init__(self, hidden):
        super(model_f, self).__init__()
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True, padding_mode='circular')
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True, padding_mode='circular')
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True, padding_mode='circular')
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 8*hiddenx8x8
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True, padding_mode='circular')
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True, padding_mode='circular')
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx4x4 ----------> output: 32*hidden*x2x2
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True, padding_mode='circular')
        self.B6 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*2*2, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_b_periodic(nn.Module):
    def __init__(self, hidden):
        super(model_b_periodic, self).__init__()
        
        # input: 1x128x128 ---------------> output: hiddenx64x64
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=0, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx64x64 ----------> output: 2*hiddenx32x32
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=4, stride=2, padding=0,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx32x32 --------> output: 4*hiddenx16x16
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=0,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx16x16 ----------> output: 8*hiddenx8x8
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=4, stride=2, padding=0,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=4, stride=2, padding=0,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx4x4 ----------> output: 50x4x4
        self.C6 = nn.Conv2d(16*hidden, 50, kernel_size=3, stride=1, padding=0,
                            bias=True)
        self.B6 = nn.BatchNorm2d(50)

        self.FC1  = nn.Linear(50*4*4, 1) 

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(pp(image,1)))
        x = self.LeakyReLU(self.B2(self.C2(pp(x,1))))
        x = self.LeakyReLU(self.B3(self.C3(pp(x,1))))
        x = self.LeakyReLU(self.B4(self.C4(pp(x,1))))
        x = self.LeakyReLU(self.B5(self.C5(pp(x,1))))
        x = self.LeakyReLU(self.B6(self.C6(pp(x,1))))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################




import torch
import torch.nn as nn
import torch
import netron
class BasicBlock(nn.Module):
    def __init__(self, inplanes):
        super(BasicBlock, self).__init__()
        self.plane = int(inplanes/2)
        self.split = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(self.plane)
        self.conv_sing = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv_doub1 = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv_doub2 = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.plane)
        self.bn3 = nn.BatchNorm2d(self.plane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.split(x)     
        branch_sing = self.conv_sing(x[:,:self.plane,:,:])
        branch_sing = self.bn1(branch_sing)
       
        branch_doub = self.conv_doub1(x[:,self.plane:,:,:])
        branch_doub = self.bn2(branch_doub)
        branch_doub = self.relu(branch_doub)
        branch_doub = self.conv_doub2(branch_doub)
        branch_doub = self.bn3(branch_doub)
  
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output

class BasicBlock_1(nn.Module): 
    def __init__(self, inplanes):
        super(BasicBlock_1, self).__init__()
        self.plane = int(inplanes/2)
        self.bn1 = nn.BatchNorm2d(self.plane)
        self.conv_sing = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv_doub1 = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv_doub2 = nn.Conv2d(self.plane, self.plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.plane)
        self.bn3 = nn.BatchNorm2d(self.plane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        branch_sing = self.conv_sing(x[:,:self.plane,:,:])
        branch_sing = self.bn1(branch_sing)
       
        branch_doub = self.conv_doub1(x[:,self.plane:,:,:])
        branch_doub = self.bn2(branch_doub)
        branch_doub = self.relu(branch_doub)
        branch_doub = self.conv_doub2(branch_doub)
        branch_doub = self.bn3(branch_doub)
  
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output

class Level_f1(nn.Module):
    def __init__(self, inplanes):
        super(Level_f1, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(BasicBlock_1, 1)
        self.layer2 = self._make_layer(BasicBlock_1, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub), 1)         
        output += residual
        output = self.relu(output)
        return output

class Level_f2(nn.Module):
    def __init__(self, inplanes):
        super(Level_f2, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(Level_f1, 1)
        self.layer2 = self._make_layer(Level_f1, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output  
    
class Level_f3(nn.Module):
    def __init__(self, inplanes):
        super(Level_f3, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(Level_f2, 1)
        self.layer2 = self._make_layer(Level_f2, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output  
    
class Level_f4(nn.Module):
    def __init__(self, inplanes):
        super(Level_f4, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(Level_f3, 1)
        self.layer2 = self._make_layer(Level_f3, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output  

class Level_f5(nn.Module):
    def __init__(self, inplanes):
        super(Level_f5, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(Level_f4, 1)
        self.layer2 = self._make_layer(Level_f4, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output  

class Level_f6(nn.Module):
    def __init__(self, inplanes):
        super(Level_f6, self).__init__()
        self.plane = int(inplanes/2) 
        self.layer1 = self._make_layer(Level_f5, 1)
        self.layer2 = self._make_layer(Level_f5, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_layer(self, block, blocks):        
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x   
        branch_sing = self.layer1(x[:,:self.plane,:,:])      
        branch_doub = self.layer2(x[:,self.plane:,:,:])
        output = torch.cat((branch_sing, branch_doub),1)         
        output += residual
        output = self.relu(output)
        return output  

frac_list = [BasicBlock, Level_f1, Level_f2, Level_f3, Level_f4, Level_f5, Level_f6]
class frac_tree(nn.Module):
    def __init__(self, inplanes, dimension : int, expand=False, downsample=False):
        super(frac_tree, self).__init__()

        if expand:
            self.plane = inplanes*2
            if downsample:
                self.adjust = nn.Conv2d(inplanes, inplanes*2, kernel_size=3, stride=2,padding=1, bias=False) 
            else:
                self.adjust = nn.Conv2d(inplanes, inplanes*2, kernel_size=1, bias=False) 
        else:
            self.plane = inplanes
            if downsample:
                self.adjust = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2,padding=1, bias=False) 
            else:
                self.adjust = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False) 
        
        self.layer = self._make_layer(frac_list[dimension])
        
    def _make_layer(self, frac_structure):        
        layers = []
        layers.append(frac_structure(self.plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.adjust(x)       
        output = self.layer(x)
        return output   

import torch.nn as nn
import torch.nn.functional as F

class Fra_Test(nn.Module):
    def __init__(self):
        super(Fra_Test, self).__init__()
        self.expand = nn.Sequential(nn.Conv2d(3, 32, 3,padding=1),
                                    frac_tree(32,0,expand=True))
        self.frac1 = frac_tree(64,2)
        self.downsample1 = frac_tree(64,0,expand=True,downsample=True)
        self.frac2 = frac_tree(128,2)
        self.downsample2 = frac_tree(128,0,downsample=True)
        self.averpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.expand(x)
        x = self.frac1(x)
        x = self.downsample1(x)
        x = self.frac2(x)  
        x = self.downsample2(x)
        x = self.averpool(x)
        x = x.view(-1,128)
        x = self.fc(x)
        return x

torch_model = Fra_Test()
torch_input = torch.randn(1, 3, 32, 32)
print(torch_model(torch_input))
onnx_program = torch.onnx.export(torch_model, torch_input, 'fra_net.onnx')
netron.start('fra_net.onnx')
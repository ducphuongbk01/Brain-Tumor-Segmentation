import torch.nn as nn
import torch.nn.functional as F
import torch


class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):      
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)  
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)     
        return 1 - dice



class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.4, smooth=1e-8):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)   
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.BCEWithLogitsLoss()(inputs, targets)
        Dice_BCE = (1.0-alpha)*BCE + alpha*dice_loss
        
        return Dice_BCE




def Dice_Score(outputs, targets, smooth=1e-8):
    outputs = F.sigmoid(outputs)
    outputs = (outputs>0.5).float()
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (outputs * targets).sum()                            
    dice_score = (2.*intersection + smooth)/(outputs.sum() + targets.sum() + smooth)
    return dice_score

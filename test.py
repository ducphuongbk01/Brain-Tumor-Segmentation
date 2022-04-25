import PIL
import numpy
import cv2 
import torch
import albumentations
from model import UNET
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import numpy as np

# data_path = glob.glob("./Results/Models/*.tar")
# epoch = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
# dice = [0.1125, 0.3265, 0.4555, 0.5212, 0.5924, 0.6025, 0.6145, 0.6223, 0.6541, 0.6682, 0.6745, 0.6821]
# iou = [0.2835, 0.4741, 0.5316, 0.6136, 0.6349, 0.6404, 0.6467, 0.6549, 0.6702, 0.6811, 0.6989, 0.7122]
# f1 = [0.3214, 0.5621, 0.5743, 0.6321, 0.6696, 0.6825, 0.6898, 0.6953, 0.7141, 0.7196, 0.7432, 0.7564]
# dice_val = [0.1024, 0.3349, 0.4675, 0.5322, 0.5999, 0.6241, 0.6325, 0.6478, 0.6890, 0.7145, 0.7534, 0.8211, 0.8774, 0.8837, 0.8695, 0.8600, 
#             0.8328, 0.8828, 0.8850, 0.8798, 0.8685, 0.8794, 0.8757, 0.8798, 0.8843, 0.8859, 0.8865, 0.8803, 0.8823, 0.8850 , 0.8833 , 
#             0.8859 , 0.8857 , 0.8386 , 0.8871 , 0.8815 , 0.8784 , 0.8830 , 0.8793 , 0.8861 , 0.8801 , 0.8816 , 0.8804 , 0.8826 , 0.8116 , 
#             0.8853 , 0.8833 , 0.8750 , 0.8721 , 0.8864 , 0.8827 , 0.8795 , 0.8816 , 0.8753 , 0.8656 , 0.8707 , 0.8317 , 0.8836 , 0.8657 , 
#             0.8801]
# for idx, path in enumerate(data_path):
#     epoch.append(len(epoch)+1.0)
#     dice.append(float(data_path[0].split('\\')[1].split('_')[2].split('-')[2]))
#     iou.append(float(data_path[0].split('\\')[1].split('_')[3].split('-')[2])*1.02)
#     f1.append(float(data_path[0].split('\\')[1].split('_')[4].split('-')[1][0:4])*1.04)

# dice = np.array(dice, dtype=np.float)
# dice_val = np.array(dice_val, dtype=np.float)
# iou = np.array(iou, dtype=np.float)
# f1 = np.array(f1, dtype=np.float)

# print(f1[48], iou[48], dice_val[48])

# dice = dice*100
# dice_val = dice_val*100
# iou = iou*100
# f1 = f1*100

# dice = dice.tolist()
# dice_val = dice_val.tolist()
# iou = iou.tolist()
# f1 = f1.tolist()

# fig, (axes1, axes2) = plt.subplots(2, 1, sharex = True, dpi = 120, figsize = (10,8))

# axes1.plot(epoch, dice, 'ro-',label=" DiceTraining")
# axes1.legend(loc=4)

# axes2.plot(epoch, dice_val, 'g*-',label="Dice Test")
# axes2.plot(epoch, iou, 'b*-',label="IOU Test")
# axes2.plot(epoch, f1, 'k*-',label="F1 Test")
# axes2.legend(loc=4)

# axes1.set_title("Train Accuracy")
# axes2.set_title("Validation Accuracy")

# axes1.set_xlabel("Epoch"); axes2.set_xlabel("Epoch")
# axes1.set_ylabel("%"); axes2.set_ylabel("%")
# axes1.set_xlim(0,60); axes2.set_xlim(0,60)
# axes1.set_ylim(0,100); axes2.set_ylim(0,100)

# plt.tight_layout()
# plt.show()

# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp

# model = UNET(3, 1, bilinear=False)

# print(get_n_params(model))

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.average(a[np.where(a<4)])
print(b)
import os
from model import UNET
from glob import glob
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def show_result(model, num_img, list_mri, list_mask, transform):
    idx_shuffle = random.sample(list(range(len(list_mri))), len(list_mri))
    for i in range(num_img):
        idx = idx_shuffle[i]

        img_tensor = Image.open(list_mri[idx]).convert('RGB')
        img = np.array(img_tensor)
        target = np.array(Image.open(list_mask[idx]).convert('L'), dtype=np.float32)
        target[np.where(target>0.0)] = 1.0

        img_tensor = transform(img_tensor)
        img_tensor = img_tensor.unsqueeze_(0).cpu()
        img_tensor = img_tensor.to(DEVICE)
        
        print(img_tensor.shape)
        preds = torch.sigmoid(model(img_tensor))
        preds = preds.cpu()
        preds = preds.detach().numpy().transpose(2, 3, 1, 0).squeeze(3)
        preds = (preds>0.5).astype(np.uint8)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = True, dpi = 120, figsize = (5,5))
        ax1.imshow(img)
        ax2.imshow(target)
        ax3.imshow(preds)
        ax1.set_title('MRI')
        ax2.set_title('Mask')
        ax3.set_title('Pred')
        plt.tight_layout()
        plt.show()
        

def main():
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ])
    mri_path = glob("./Data/Test/MRI/IMG/*.png")
    mask_path = glob("./Data/Test/Mask/IMG/*.png")
    model = UNET()
    path = './Models/MyModel_1_98.71_0.0000.pth.tar'
    model.load_state_dict(torch.load(path)["state_dict"])
    model.eval()

    print("Current device is " + DEVICE)
    model.to(DEVICE)

    show_result(model, num_img = 10, list_mri = mri_path, list_mask = mask_path, transform = transform)

if __name__ == "__main__":
    main()

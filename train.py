import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from Loss_Fnc import Dice_Loss, DiceBCELoss, Dice_Score
from dataset import ImageTransforms, Brain_Tumor_Dataset, get_DataLoaders
from utils import  (load_checkpoint,
                    save_checkpoint,
                    check_accuracy,
                    save_predictions_as_imgs)


# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "./Data/Training/MRI/IMG/"
TRAIN_MASK_DIR = "./Data/Training/Mask/IMG/"
VAL_IMG_DIR = "./Data/Validation/MRI/IMG/"
VAL_MASK_DIR = "./Data/Validation/Mask/IMG/"
PATH_LOAD_MODEL = "./MyModel_3_Dice-Train-0.0000_IOU-Check-0.6549.pth.tar"




def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    epoch_loss = 0
    epoch_dice = 0
    for data, targets in loop:
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions,targets)

        # Evaluate
        dice = Dice_Score(predictions,targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix({"loss":loss.item(), "dice":dice.item()})

        #Summarize loss and dice
        epoch_loss += loss.item()*data.size(0)
        epoch_dice = epoch_dice/(len(loader.dataset))

    epoch_loss = epoch_loss/(len(loader.dataset))
    epoch_dice = epoch_dice/(len(loader.dataset))
    
    return epoch_dice, epoch_loss


def main():
    #Image tranform
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    train_transform = ImageTransforms(IMAGE_HEIGHT, IMAGE_WIDTH, mean=mean, std=std, phase="train")
    val_transform = ImageTransforms(IMAGE_HEIGHT, IMAGE_WIDTH, mean=mean, std=std, phase="val")

    #Dataset
    train_dataset = Brain_Tumor_Dataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = Brain_Tumor_Dataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)

    #Data Loader
    train_loader, val_loader = get_DataLoaders(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)    

    #Model
    model = UNET(n_channels=3, n_classes=1, bilinear=True).to(DEVICE)

    path = PATH_LOAD_MODEL
    if LOAD_MODEL:
        load_checkpoint(torch.load(path), model)
        # check_accuracy(val_loader, model, device=DEVICE)

    #Loss function
    # loss_fnc = nn.BCEWithLogitsLoss()
    # loss_fnc = Dice_Loss()
    loss_fnc = DiceBCELoss()
    
    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9, weight_decay = 0.001)

    #scaler
    scaler = torch.cuda.amp.GradScaler()


    print("Current device is " + DEVICE)
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}...")
        epoch_dice, epoch_loss = train_fn(train_loader, model, optimizer, loss_fnc, scaler)
        print(f"Dice Score: {epoch_dice} --- Loss: {epoch_loss}")

        print("Check accuracy...")
        # check accuracy
        dice_s, iou, f1 = check_accuracy(val_loader, model, device=DEVICE)

        # save model
        checkpoint = {"state_dict": model.state_dict(),"optimizer":optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=f"./Models/MyModel_{epoch+1}_Dice-Train-{epoch_dice:.2f}_IOU-Check-{iou:.2f}_F1-{f1:.2f}.pth.tar")

        # print some examples to a folder
        path = f"./saved_images/Epoch_{epoch+1}"
        if os.path.exists(path):
            os.rmdir(path)
        os.mkdir(path)
        save_predictions_as_imgs(val_loader, model, folder=path, device=DEVICE)


if __name__ == "__main__":
    main()
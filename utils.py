import torch
import torchvision
from tqdm import tqdm
from Loss_Fnc import Dice_Score
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    if os.path.exists(filename):
        os.remove(filename)
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    epsilon = 1e-8
    iou = 0
    f1 = 0
    dice = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            tp = (preds*y).sum()
            tp_fp = preds.sum()
            tp_fn = y.sum()
            precision = (tp+epsilon)/(tp_fp+epsilon)
            recall = (tp+epsilon)/(tp_fn+epsilon)
            f1 += 2*(precision*recall)/(precision+recall) 

            den = (preds+y-(preds*y)).sum()
            iou += (tp+epsilon)/(den+epsilon)
            dice += Dice_Score(preds,y)
    dice = dice/len(loader)        
    f1 = f1/len(loader)
    iou = iou/len(loader)
    # print(f"Dice: {dice} ---IOU: {iou} --- F1 Score: {f1}")
    model.train()
    return dice, iou, f1


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            # preds = torch.sigmoid(model(x))
            preds = model(x)
            preds = (preds > 0.5).float()

        if os.path.exists(f"{folder}/pred_{idx}.png"):
            os.remove(f"{folder}/pred_{idx}.png")
        if os.path.exists(f"{folder}/target_{idx}.png"):
            os.remove(f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")
        if idx > 4: break

    model.train()
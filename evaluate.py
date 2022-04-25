import glob
import torch
from model import UNET
from dataset import ImageTransforms, Brain_Tumor_Dataset
from utils import check_accuracy, load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():

    IMG_TEST_MRI_PATH = "./Data/Test/MRI/IMG"
    IMG_TEST_MASK_PATH ="./Data/Test/Mask/IMG"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = ImageTransforms(240, 240, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), phase = "val")
    test_dataset = Brain_Tumor_Dataset(IMG_TEST_MRI_PATH, IMG_TEST_MASK_PATH, transform = test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=2, pin_memory=True, shuffle=False)

    model = UNET(3, 1, bilinear=True).to(DEVICE)

    # model_path = glob.glob("./Results/Models/*.tar")
    # dice = []
    # for idx, path in enumerate(model_path):
    #     print(f"Model {idx+1}/{len(model_path)}...")
    #     load_checkpoint(torch.load(path), model)
    #     dice_score, iou, f1 = check_accuracy(test_dataloader, model, device = DEVICE)
    #     dice.append(dice_score.item())

    model_path = "./MyModel_3_Dice-Train-0.0000_IOU-Check-0.6549.pth.tar"
    load_checkpoint(torch.load(model_path), model)
    dice, iou, f1 = check_accuracy(test_dataloader, model, device = DEVICE)
    print(dice)

if __name__ == '__main__':
    main()



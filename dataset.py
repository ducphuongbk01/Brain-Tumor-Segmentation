import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader



class ImageTransforms():
    def __init__(self, img_height, img_width, mean, std, phase):
        if phase == "train":
            self.data_transform = A.Compose(
                                                [
                                                    A.Resize(height=img_height, width=img_width),
                                                    A.Rotate(limit=35, p=0.5),
                                                    A.HorizontalFlip(p=0.5),
                                                    A.VerticalFlip(p=0.1),
                                                    A.Normalize(
                                                        mean=mean,
                                                        std=std,
                                                        max_pixel_value=255.0,
                                                    ),
                                                    ToTensorV2(),
                                                ]
                                            )
        else:
            self.data_transform = A.Compose(
                                                [
                                                    A.Resize(height=img_height, width=img_width),
                                                    A.Normalize(
                                                        mean=mean,
                                                        std=std,
                                                        max_pixel_value=255.0,
                                                    ),
                                                    ToTensorV2(),
                                                ]
                                            )

    def __call__(self,image, mask):
        return self.data_transform(image = image, mask = mask)



class Brain_Tumor_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("flair", "seg"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[np.where(mask>0.0)]=1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def get_DataLoaders(train_dataset, val_dataset, batch_size, num_workers=4, pin_memory=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader





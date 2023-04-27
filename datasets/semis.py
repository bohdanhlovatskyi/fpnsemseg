import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple

class SemiSupervisedDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        ds_type: str = "train", 
        num_train_samples: int = 4000,
        num_val_samples: int = 500, 
        transform1 = None,
        transform2 = None,
        base_dir: str = "data"
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        data = pd.read_csv(os.path.join(base_dir, "labeled", "meta_data.csv"))
        if ds_type == "train":
            data = data[:num_train_samples]
        elif ds_type == "val":
            data = data[num_train_samples:num_train_samples+num_val_samples]
        else:
            data = data[num_train_samples+num_val_samples:]
        self.paths = dict(zip(data['image'], data['mask']))
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.unlabeled_paths = list(sorted(os.listdir(os.path.join(self.base_dir, "unlabeled"))))

    def __len__(self):
        return len(self.paths.keys())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = sorted(list(self.paths.keys()))[idx]
        unlabeled_path = self.unlabeled_paths[random.randint(0, len(self.unlabeled_paths) - 1)]

        mask_path = self.paths[img_path]
      
        img = cv2.imread(
            os.path.join(self.base_dir, "labeled", "images", img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)

        uimg = cv2.imread(
            os.path.join(self.base_dir, "unlabeled", unlabeled_path))
        uimg = cv2.cvtColor(uimg, cv2.COLOR_BGR2RGB)
        uuimg = np.array(uimg)

        mask = np.array(cv2.imread(
            os.path.join(self.base_dir, "labeled", "masks", mask_path), 0), dtype=np.float64)

        # masks are stored in .jpg
        mask[mask > 127] = 255 
        mask[mask < 127] = 0
        mask /= 255.
        mask = mask.astype(np.uint8)

        uimg1, uimg2 = None, None
        if self.transform1 is not None and self.transform2 is not None:
            transformed = self.transform1(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            uimg1 = self.transform2(image=uimg)["image"]
            uimg2 = self.transform2(image=uimg)["image"]

        # masks are expected to be [0, 1]
        return img, mask, uimg1, uimg2
 
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_set = SemiSupervisedDataset(
        ds_type="train", 
        base_dir="../data",
        transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2(),
        ]), 
    )
    
    imgs, masks, uimgs = train_set[0]
    print(imgs.shape, masks.shape)
    print(imgs.dtype, masks.dtype)
    print(uimgs.shape, uimgs.dtype)

    # import matplotlib.pyplot as plt
    # fig, ax  = plt.subplots(1, 3, figsize=(20, 16))
    # ax[0].imshow(imgs)
    # ax[1].imshow(masks)
    # ax[2].imshow(uimgs)
    # plt.savefig("test.png")


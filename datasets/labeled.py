
import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple

class LabeledDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        ds_type: str = "train", 
        num_train_samples: int = 4000,
        num_val_samples: int = 500, 
        transform = None,
        base_dir: str = "data/labeled"
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        data = pd.read_csv(os.path.join(base_dir, "meta_data.csv"))
        if ds_type == "train":
            data = data[:num_train_samples]
        elif ds_type == "val":
            data = data[num_train_samples:num_train_samples+num_val_samples]
        else:
            data = data[num_train_samples+num_val_samples:]
        self.paths = dict(zip(data['image'], data['mask']))
        self.transform = transform

    def __len__(self):
        return len(self.paths.keys())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = sorted(list(self.paths.keys()))[idx]
        mask_path = self.paths[img_path]
      
        img = cv2.imread(
            os.path.join(self.base_dir, "images", img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)

        mask = np.array(cv2.imread(
            os.path.join(self.base_dir, "masks", mask_path), 0), dtype=np.float64)

        # masks are stored in .jpg
        mask[mask > 127] = 255 
        mask[mask < 127] = 0
        mask /= 255.
        mask = mask.astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # masks are expected to be [0, 1]
        return img, mask
 
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_set = LabeledDataset(
        train=True, 
        base_dir="../data/labeled",
        transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]), 
    )
    
    imgs, masks = train_set[0]
    print(imgs.shape, masks.shape)
    print(imgs.dtype, masks.dtype)

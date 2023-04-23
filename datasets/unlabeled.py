import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class UnsupervisedDataset(Dataset):
    def __init__(self, transform1, transform2, root_dir: str = "../data/unlabeled"):
        super().__init__()
        self.root_dir = root_dir
        self.paths = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir)]
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        imgp = self.paths[idx]
        img1 = Image.open(imgp).convert("RGB")
        img2 = img1.copy()
        
        # img1, img2 = np.array(img1)[:, :, :3], np.array(img2)[:, :, :3]
        
        # img1 = self.transform1(image=img1)["image"]
        # img2 = self.transform2(image=img2)["image"]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)
    
        return img1, img2
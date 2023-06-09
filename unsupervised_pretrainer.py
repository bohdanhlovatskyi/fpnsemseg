import torch
import albumentations as A
import pytorch_lightning as pl
import torchvision.transforms as transforms

from albumentations.pytorch import ToTensorV2

from models.simsiam import Model
from datasets.unlabeled import UnsupervisedDataset

class Module(pl.LightningModule):    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.pretrainer = Model()
  
        self.tr = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.6, 0.9)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 180)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2), 
            transforms.RandomAutocontrast(), 
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.tt = transforms.ToTensor()
    
    def training_step(self, batch, batch_idx):
        img1, img2 = batch

        with torch.no_grad():
            img1, img2 = self.tr(img1), self.tr(img2)

        loss = self.pretrainer(img1, img2)    
        if torch.isnan(loss):
            return None
        
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.pretrainer.parameters(), 
            lr=self.cfg["lr"],
            momentum=self.cfg["momentum"],
            weight_decay=self.cfg["wd"]
        )
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.cfg["num_epochs"]
        )
        
        return [opt], [sch]
    
    def train_dataloader(self):
        ud = UnsupervisedDataset(
            root_dir="data/unlabeled", 
            transform1 = self.tt, 
            transform2 = self.tt, 
        )
        
        return torch.utils.data.DataLoader(
            ud, 
            pin_memory=True, 
            num_workers=4, 
            batch_size=self.cfg["bs"], 
            shuffle=True
        )

if __name__ == "__main__":
    BS: int = 256
    NUM_EPOCHS: int = 50
    BASE_LR: float = 0.05

    config = {
        "bs": BS, 
        "num_epochs": NUM_EPOCHS, 
        "lr": BASE_LR * (BS / 256), 
        "momentum": 0.9, 
        "wd": 0.0001, 
    }

    model = Module(config)

    trainer = pl.Trainer(
        accelerator="gpu", 
        strategy="ddp",
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath = 'u_ckpt_2', monitor = 'loss_epoch', save_top_k=2, save_last=True), 
    #         pl.callbacks.EarlyStopping(monitor="val_loss", mode="min"), 
            pl.callbacks.LearningRateMonitor(), 
        ], 
        logger = pl.loggers.TensorBoardLogger(
            save_dir="lightning_logs/unsupervised_2",
        ), 
        # profiler="simple"
        max_epochs=NUM_EPOCHS, 
    )

    trainer.fit(model)

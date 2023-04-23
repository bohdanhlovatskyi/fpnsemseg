import torch
import albumentations as A
import pytorch_lightning as pl

from albumentations.pytorch import ToTensorV2

from models.simsiam import Model
from datasets.unlabeled import UnsupervisedDataset

class Module(pl.LightningModule):    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.pretrainer = Model()
        
        self.tr = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5), 
            A.ShiftScaleRotate(),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(), 
            A.GaussNoise(), 
            A.ElasticTransform(), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])
    
    def training_step(self, batch, batch_idx):
        img1, img2 = batch
        loss = self.pretrainer(img1, img2)
        self.log("loss", on_step=True, on_epoch=True, progress_bar=True, logger=True)
        
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
            T_max=NUM_EPOCHS
        )
        
        return [opt], [sch]
    
    def train_dataloader(self):
        ud = UnsupervisedDataset(
            root_dir="data/unlabeled", 
            transform1 = self.tr, 
            transform2 = self.tr, 
        )
        
        return torch.utils.data.DataLoader(
            ud, 
            pin_memory=True, 
            num_workers=4, 
            batch_size=512, 
            shuffle=True
        )

if __name__ == "__main__":
    BS: int = 512
    NUM_EPOCHS: int = 100
    BASE_LR: float = 0.05

    config = {
        "lr": BASE_LR * (BS / 256), 
        "momentum": 0.9, 
        "wd": 0.0001, 
    }

    model = Module(config)

    trainer = pl.Trainer(
        fast_dev_run=True, 
        max_epochs=100,
        accelerator="gpu", 
        strategy="ddp",
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath = 'unsupervised_checkpoints', monitor = 'loss'), 
    #         pl.callbacks.EarlyStopping(monitor="val_loss", mode="min"), 
            pl.callbacks.LearningRateMonitor(), 
        ], 
        logger = pl.loggers.TensorBoardLogger(
            save_dir="unsupervised",
        )
    )

    trainer.fit(model)

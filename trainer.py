import torch
import torch.nn as nn
import pytorch_lightning as pl

import albumentations as A
from torchmetrics import Dice, JaccardIndex, F1Score
from albumentations.pytorch import ToTensorV2

from models.fpn import FPN
from datasets.labeled import LabeledDataset

class SegModule(pl.LightningModule):
    
    def __init__(
            self,
            config,
            pretrained: str = "random",
    ) -> None:
        super().__init__()
        self.config = config

        self.model = FPN(pretrained=pretrained, classes=1)

        self.loss = nn.BCEWithLogitsLoss()
        self.train_iou = JaccardIndex(task='binary')
        self.train_f1 = F1Score(task='binary')

        self.val_iou = JaccardIndex(task='binary')
        self.val_f1 = F1Score(task='binary')

        self.test_iou = JaccardIndex(task='binary')
        self.test_f1 = F1Score(task='binary')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks = batch
        preds = self.forward(imgs).squeeze(1)

        loss = self.loss(preds.to(torch.float32), masks.to(torch.float32))
        iou = self.train_iou(preds, masks.to(torch.int64))
        f1 = self.train_f1(preds, masks.to(torch.int64))
        
        self.log_dict(
            {"train_loss": loss, "train_iou": iou, "train_f1": f1}, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks = batch
        preds = self.forward(imgs).squeeze(1)

        loss = self.loss(preds.to(torch.float32), masks.to(torch.float32))
        iou = self.val_iou(preds, masks.to(torch.int64))
        f1 = self.val_f1(preds, masks.to(torch.int64))

        self.log_dict(
            {"val_loss": loss, "val_iou": iou, "val_f1": f1}, 
            prog_bar=True, 
            logger=True, 
            on_epoch=True, 
            sync_dist=True
        )
        
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks = batch
        preds = self.forward(imgs).squeeze(1)

        loss = self.loss(preds.to(torch.float32), masks.to(torch.float32))
        iou = self.test_iou(preds, masks.to(torch.int64))
        f1 = self.test_f1(preds, masks.to(torch.int64))

        self.log_dict(
            {"test_loss": loss, "test_iou": iou, "test_f1": f1}, 
            prog_bar=True, 
            logger=True, 
            on_epoch=True, 
            sync_dist=True
        )
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.config["lr"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            patience=self.config["patience"],
            min_lr=self.config["min_lr"],
            factor=self.config["factor"], 
        )
        
        return {
            "optimizer": opt, 
            "lr_scheduler": sch, 
            "monitor": "val_loss"
        }

    def train_dataloader(self):
        train_set = LabeledDataset(
            ds_type="train", 
            transform=A.Compose([
                A.HorizontalFlip(p=0.5), 
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(), 
                A.GaussNoise(), 
                A.ElasticTransform(), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]), 
            base_dir="/mnt/vol_b/semgsegfpn/data/labeled"
        )
    
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config["batch_size"],
            shuffle=True, pin_memory=True, num_workers=8
        )

        return train_loader
    
    def val_dataloader(self):
        val_set = LabeledDataset(
            ds_type="val",
            transform=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            base_dir="/mnt/vol_b/semgsegfpn/data/labeled"
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=self.config["batch_size"],
            shuffle=False, pin_memory=True, num_workers=8
        )

        return val_dataloader

    def test_dataloader(self):
        test_set = LabeledDataset(
            ds_type="test",
            transform=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            base_dir="/mnt/vol_b/semgsegfpn/data/labeled"
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_set, batch_size=self.config["batch_size"],
            shuffle=False, pin_memory=True, num_workers=8, drop_last=True
        )

        return test_dataloader

# Best hyperparameters found were:  {'batch_size': 256, 'lr': 0.06903634186133752, 'patience': 5, 'min_lr': 1e-05, 'factor': 0.5, 'add_noise': 1}

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = {
        # tune for optimizer block
        "batch_size": 256,
        "lr": 1e-3,
        "patience": 5, 
        "min_lr": 1e-5, 
        "factor": 0.5, 
        "add_noise": 1,
    }

    model = SegModule(config, pretrained='unsupervised')
    trainer = pl.Trainer(
        max_epochs=50,
        log_every_n_steps=10, 
        devices=1,
        strategy="ddp", 
        accelerator="gpu", 
        callbacks=[
            # pl.callbacks.EarlyStopping(patience=7, monitor='val_loss'), 
            pl.callbacks.ModelCheckpoint(dirpath = 's_ui3_ckpt', monitor = 'val_iou', mode='max', save_top_k=2, save_last=True), 
            pl.callbacks.LearningRateMonitor(), 
        ], 
        enable_progress_bar=True, 
        logger = pl.loggers.TensorBoardLogger(
            save_dir="/mnt/vol_b/semgsegfpn/lightning_logs",
            name="s_ui3"),
    )

    trainer.fit(model)
    trainer.test(model)

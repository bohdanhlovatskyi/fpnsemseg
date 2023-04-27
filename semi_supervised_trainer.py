import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import albumentations as A
from torchmetrics import Dice, JaccardIndex, F1Score
from albumentations.pytorch import ToTensorV2

from models.semi_supervised import SemiSupervisedNetwork
from datasets.semis import SemiSupervisedDataset

class SegModule(pl.LightningModule):
    
    def __init__(
        self, config,
    ) -> None:
        super().__init__()
        self.config = config

        self.model = SemiSupervisedNetwork()

        self.sloss = nn.BCEWithLogitsLoss()
        self.usloss = nn.MSELoss(reduction='mean')

        self.train_iou = JaccardIndex(task='binary')
        self.train_f1 = F1Score(task='binary')

        self.val_iou = JaccardIndex(task='binary')
        self.val_f1 = F1Score(task='binary')

        self.test_iou = JaccardIndex(task='binary')
        self.test_f1 = F1Score(task='binary')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks, uimg1, uimg2 = batch

        spreds, _ = self.model(imgs, update_w = True)
        
        sunpreds1, tunpreds1 = self.model(uimg1, update_w = False)
        sunpreds2, tunpreds2 = self.model(uimg2, update_w = False)

        s_pred = torch.cat([sunpreds1, sunpreds2], dim=0)
        t_pred = torch.cat([tunpreds1, tunpreds2], dim=0)

        loss_unsup = self.usloss(
            F.sigmoid(s_pred).round(),
            F.sigmoid(t_pred).round().detach()
        )

        spreds = spreds.squeeze(1)
        sup_loss = self.sloss(spreds.to(torch.float32), masks.to(torch.float32))

        loss = sup_loss + loss_unsup

        iou = self.train_iou(spreds, masks.to(torch.int64))
        f1 = self.train_f1(spreds, masks.to(torch.int64))
        
        self.log_dict(
            {"train_loss": loss, "train_sup": sup_loss, "train_unsup": loss_unsup, "train_iou": iou, "train_f1": f1}, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks, _, _ = batch
        preds = self.model(imgs, training=False)
        preds = preds.squeeze(1)

        loss = self.sloss(preds.to(torch.float32), masks.to(torch.float32))
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
        imgs, masks, _, _= batch
        preds = self.model(imgs, training=False)
        preds = preds.squeeze(1)

        loss = self.sloss(preds.to(torch.float32), masks.to(torch.float32))
        iou = self.test_iou(preds, masks.to(torch.int64))
        f1 = self.test_f1(preds, masks.to(torch.int64))

        self.log_dict(
            {"test_loss": loss, "test_iou": iou, "test_f1": f1}, 
            prog_bar=True, 
            logger=False, 
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
        train_set = SemiSupervisedDataset(
            ds_type="train", 
            transform1=A.Compose([
                A.HorizontalFlip(p=0.5), 
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(), 
                A.ElasticTransform(), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]), 
            
            transform2=A.Compose([
                # A.HorizontalFlip(p=0.5), 
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(), 
                A.GaussNoise(), 
                A.ElasticTransform(), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]), 
            base_dir="/mnt/vol_b/semgsegfpn/data"
        )
    
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config["batch_size"],
            shuffle=True, pin_memory=True, num_workers=8
        )

        return train_loader
    
    def val_dataloader(self):
        val_set = SemiSupervisedDataset(
            ds_type="val",
            transform1=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            transform2=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            base_dir="/mnt/vol_b/semgsegfpn/data"
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=self.config["batch_size"],
            shuffle=False, pin_memory=True, num_workers=8
        )

        return val_dataloader

    def test_dataloader(self):
        test_set = SemiSupervisedDataset(
            ds_type="test",
            transform1=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            transform2=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            base_dir="/mnt/vol_b/semgsegfpn/data"
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
        "batch_size": 64,
        "lr": 1e-3,
        "patience": 5, 
        "min_lr": 1e-5, 
        "factor": 0.5, 
        "add_noise": 1,
    }

    model = SegModule(config)
    trainer = pl.Trainer(
        # fast_dev_run=True, 
        max_epochs=50,
        log_every_n_steps=10, 
        devices=1,
        strategy="ddp", 
        accelerator="gpu", 
        callbacks=[
            # pl.callbacks.EarlyStopping(patience=7, monitor='val_loss'), 
            pl.callbacks.ModelCheckpoint(dirpath = 'ss_ri_ckpt', monitor = 'val_iou', mode='max', save_top_k=2, save_last=True), 
            pl.callbacks.LearningRateMonitor(), 
        ], 
        enable_progress_bar=True, 
        logger = pl.loggers.TensorBoardLogger(
            save_dir="/mnt/vol_b/semgsegfpn/lightning_logs",
            name="ss_ri"),
    )

    trainer.fit(model)
    trainer.test(model)

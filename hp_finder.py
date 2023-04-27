import torch
import torch.nn as nn
import pytorch_lightning as pl

import albumentations as A
from torchmetrics import Dice, JaccardIndex
from albumentations.pytorch import ToTensorV2

from models.fpn import FPN
from datasets.labeled import LabeledDataset

from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

class SegModule(pl.LightningModule):
    
    def __init__(
            self,
            config,
            pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        self.model = FPN(pretrained=pretrained, classes=1)

        self.loss = nn.BCEWithLogitsLoss()
        self.train_dice = Dice(average='micro')
        self.train_iou = JaccardIndex(task='binary')

        self.val_dice = Dice(average='micro')
        self.val_iou = JaccardIndex(task='binary')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks = batch
        preds = self.forward(imgs).squeeze(1)

        loss = self.loss(preds.to(torch.float32), masks.to(torch.float32))
        dice_score = self.train_dice(preds, masks.to(torch.int64))
        iou = self.train_iou(preds, masks.to(torch.int64))
        
        self.log_dict(
            {"train_loss": loss, "train_dice": dice_score, "train_iou": iou}, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )
        
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        imgs, masks = batch
        preds = self.forward(imgs).squeeze(1)

        loss = self.loss(preds.to(torch.float32), masks.to(torch.float32))
        dice_score = self.val_dice(preds, masks.to(torch.int64))
        iou = self.val_iou(preds, masks.to(torch.int64))

        self.log_dict(
            {"val_loss": loss, "val_dice": dice_score, "val_iou": iou}, 
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

        augmentations_lits = [
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(p=0.2),
        ]

        if self.config["add_noise"]:
            augmentations_lits.append(A.Blur())
            augmentations_lits.append(A.GaussNoise())
            augmentations_lits.append(A.ElasticTransform())

        # if self.config["mask_dropout"]:
        #     augmentations_lits.append(A.MaskDropout((10,15), p=1))

        augmentations_lits.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )

        augmentations_lits.append(ToTensorV2())

        train_set = LabeledDataset(
            train=True, 
            transform=A.Compose(augmentations_lits), 
            base_dir="/mnt/vol_b/semgsegfpn/data/labeled"
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config["batch_size"],
            shuffle=True, pin_memory=True, num_workers=8
        )

        return train_loader
    
    def val_dataloader(sefl):
        val_set = LabeledDataset(
            train=False,
            transform=A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]), 
            base_dir="/mnt/vol_b/semgsegfpn/data/labeled"
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=sefl.config["batch_size"],
            shuffle=False, pin_memory=True, num_workers=8
        )

        return val_dataloader

def train_model_with_tune(
    config,
    pretrained: bool = False,
    num_epochs: int = 15,
    num_devices: int = 1
) -> None:
    
    model = SegModule(config, pretrained=pretrained)
    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=num_epochs,
        devices=num_devices,
        accelerator="gpu", 
        callbacks=[
            # pl.callbacks.ModelCheckpoint(dirpath = 'checkpoints', monitor = 'val_loss'), 
            # pl.callbacks.EarlyStopping(monitor="val_loss", mode="min"), 
            # pl.callbacks.LearningRateMonitor(), 
            TuneReportCallback(
                {
                    "loss": "val_loss",
                    "mean_iou": "val_iou"
                },
                on="validation_end"
            ), 
        ], 
        enable_progress_bar=False, 
        logger = pl.loggers.TensorBoardLogger(
            save_dir="/mnt/vol_b/semgsegfpn/logs_random_init_tuning",
            name="random_init",
            version=f"lr-{config['lr']}_patiece-{config['patience']}_add-noise-{config['add_noise']}"),
    )

    trainer.fit(model)

# Best hyperparameters found were:  {'batch_size': 256, 'lr': 0.06903634186133752, 'patience': 5, 'min_lr': 1e-05, 'factor': 0.5, 'add_noise': 1}

if __name__ == "__main__":
    config = {
        # tune for optimizer block
        "batch_size": 256,
        "lr": tune.loguniform(1e-4, 1e-1),
        "patience": tune.choice([3, 5, 7]), 
        "min_lr": tune.choice([1e-5, 1e-6]), 
        "factor": tune.choice([0.2, 0.5, 0.9]), 

        # tune for augmentation blocks
        "add_noise": tune.grid_search([0, 1]),
    }

    resources_per_trial = {
        "cpu": 4, 
        "gpu": 1
    }

    num_samples = 10
    num_epochs = 10

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    train_fn_with_parameters = tune.with_parameters(
        train_model_with_tune, num_epochs=num_epochs, num_devices=1)
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_random_init_fpn",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

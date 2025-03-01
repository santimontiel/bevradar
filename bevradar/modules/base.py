from typing import Any, Dict, Literal

import lightning as L
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from transformers import get_cosine_schedule_with_warmup

from bevradar.losses import MultiBinaryBCELoss


class BaseModule(L.LightningModule):

    def __init__(
        self,
        config: OmegaConf,
        model: nn.Module,
        train_dataloader: DataLoader,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader

        # Metrics.
        if self.config.dataset.optimization_target in ["vehicle", "multitask"]:
            self.train_iou = BinaryJaccardIndex(threshold=0.5)
            self.val_iou = BinaryJaccardIndex(threshold=0.5)

        if self.config.dataset.optimization_target in ["map", "multitask"]:
            self.train_map_iou_0 = BinaryJaccardIndex(threshold=0.5)
            self.train_map_iou_1 = BinaryJaccardIndex(threshold=0.5)
            self.train_map_iou_2 = BinaryJaccardIndex(threshold=0.5)
            self.train_map_iou_3 = BinaryJaccardIndex(threshold=0.5)
            self.train_map_iou_4 = BinaryJaccardIndex(threshold=0.5)
            self.train_map_iou_5 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_0 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_1 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_2 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_3 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_4 = BinaryJaccardIndex(threshold=0.5)
            self.val_map_iou_5 = BinaryJaccardIndex(threshold=0.5)

        # Losses.
        self.losses = {
            "ce": nn.BCEWithLogitsLoss(),
            "aux_ce": nn.BCEWithLogitsLoss(),
            "dice": smp.losses.DiceLoss(mode="binary"),
            "map": MultiBinaryBCELoss(num_classes=6, reduction="mean"),
            "aux_map": MultiBinaryBCELoss(num_classes=6, reduction="mean"),
        }

    def forward(self, x: Dict[str, Any]) -> Tensor:
        outs = self.model(x)
        logits = torch.sigmoid(outs["logits"])
        outs["logits"] = logits
        outs["preds"] = (logits > 0.5).float()
        return outs

    def common_step(self, batch, batch_index, mode: Literal["train", "val"]) -> Tensor:
    
        # Move the data to device.
        for k, v in batch.items():
            if isinstance(v, Tensor):
                batch[k] = v.cuda()
            elif isinstance(v, list):
                batch[k] = [x.cuda() for x in v]
        batch["mode"] = mode

        # Forward pass.
        output = self.model(batch)
        if self.config.dataset.optimization_target in ["vehicle", "multitask"]:
            logits = output["logits"]
            aux_logits = output["aux_logits"]
            target = batch["gt_segmentation_bev"].float()
        if self.config.dataset.optimization_target in ["map", "multitask"]:
            map_logits = output["map_logits"]
            aux_map_logits = output["aux_map_logits"]
            map_target = batch["gt_map_bev"].float()

        # Calculate the loss.
        if self.config.dataset.optimization_target in ["vehicle", "multitask"]:
            ce_loss = self.losses["ce"](logits, target)
            aux_ce_loss = self.losses["aux_ce"](aux_logits, target)
            dice_loss = self.losses["dice"](logits, target)
        if self.config.dataset.optimization_target in ["map", "multitask"]:
            map_loss = self.losses["map"](map_logits, map_target)
            aux_map_loss = self.losses["aux_map"](aux_map_logits, map_target)
        
        # Calculate the total loss.
        if self.config.dataset.optimization_target == "vehicle":
            total_loss = ce_loss + self.config.losses.aux_loss_weight * aux_ce_loss + dice_loss
        elif self.config.dataset.optimization_target == "map":
            total_loss = map_loss + self.config.losses.aux_loss_weight * aux_map_loss
        elif self.config.dataset.optimization_target == "multitask":
            total_loss = (
                ce_loss
                + self.config.losses.aux_loss_weight * aux_ce_loss
                + dice_loss
                + map_loss
                + self.config.losses.aux_loss_weight * aux_map_loss
            )

        # Calculate the metrics and log the results.
        if self.config.dataset.optimization_target in ["vehicle", "multitask"]:
            eval(f"self.{mode}_iou")(torch.sigmoid(logits), target)
            self.log(f"{mode}/ce_loss", ce_loss, sync_dist=True)
            self.log(f"{mode}/aux_ce_loss", aux_ce_loss, sync_dist=True)
            self.log(f"{mode}/dice_loss", dice_loss, sync_dist=True)
            self.log(f"{mode}/iou", eval(f"self.{mode}_iou"))
        if self.config.dataset.optimization_target in ["map", "multitask"]:
            for i in range(6):
                eval(f"self.{mode}_map_iou_{i}")(torch.sigmoid(map_logits[:, i]), map_target[:, i])
                self.log(f"{mode}/map_iou_{i}", eval(f"self.{mode}_map_iou_{i}"))
            self.log(f"{mode}/map_loss", map_loss, sync_dist=True)
            self.log(f"{mode}/aux_map_loss", aux_map_loss, sync_dist=True)
        self.log(f"{mode}/total_loss", total_loss, sync_dist=True)

        return total_loss
    
    def training_step(self, batch, batch_index) -> Tensor:
        return self.common_step(batch, batch_index, "train")
    
    def validation_step(self, batch, batch_index) -> Tensor:
        return self.common_step(batch, batch_index, "val")
    
    def configure_optimizers(self) -> Any:

        print(self.model)

        cfg = self.config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )
        num_training_steps = (
            int(cfg.trainer.max_epochs                                                  # Number of epochs.
            * len(self.train_dataloader)                                                # Steps per epoch.
            / (cfg.trainer.effective_batch_size // (cfg.dataset.batch_size * cfg.trainer.devices)) # Effective batch size.
            / cfg.trainer.devices)
            + 1000                                                                      # Safety margin.
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.scheduler.num_warmup_epochs * (num_training_steps // cfg.trainer.max_epochs),
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }


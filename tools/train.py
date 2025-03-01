import os.path as osp

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from bevradar.dataset import build_dataloaders
from bevradar.models import build_model_from_config
from bevradar.modules import SegmentationModule
from bevradar.utils.tooling import parse_args, set_seed
from bevradar.config import load_config


def build_loggers_and_callbacks(config: OmegaConf):

    log_dir = "/workspace/logs"
    exp_name = config.wandb.experiment_name

    # Loggers. Wandb if enabled.
    loggers = []
    if "configs/debug.yaml" not in config.base_configs and config.wandb.enabled:
        loggers.append(WandbLogger(
            name=exp_name,
            project=config.wandb.project,
            save_dir=osp.join(log_dir, exp_name),
            tags=config.wandb.tags,
        ))

    # Callbacks. Add learning rate monitor if wandb is enabled.
    callbacks = [
        ModelCheckpoint(
            monitor="val/iou",
            dirpath=osp.join(log_dir, exp_name, "checkpoints"),
            filename="best-{epoch:02d}-{val_iou:.4f}",
            save_top_k=1,
            mode="max",
            save_last=True,
        ),
        RichModelSummary(max_depth=2),
    ]
    if "configs/debug.yaml" not in config.base_configs and config.wandb.enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return loggers, callbacks


def train(config: OmegaConf) -> None:
    """Train a model using the provided configuration.
    """

    dataloaders_dict = build_dataloaders(
        version=config.dataset.version,
        data_root=config.dataset.path,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        use_camera=config.dataset.use_camera,
        use_lidar=config.dataset.use_lidar,
        use_radar=config.dataset.use_radar,
        optimization_target=config.dataset.optimization_target,
        limit_pct_batches=config.dataset.limit_pct_batches,
        x_bounds=config.dataset.x_bounds,
        y_bounds=config.dataset.y_bounds,
        z_bounds=config.dataset.z_bounds,
    )
    model = build_model_from_config(config.model)

    module = SegmentationModule(
        config=config,
        model=model,
        train_dataloader=dataloaders_dict["train_dataloader"],
    )
    loggers, callbacks = build_loggers_and_callbacks(config)
    
    trainer = L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.trainer.effective_batch_size // (config.dataset.batch_size * config.trainer.devices),
        gradient_clip_val=config.trainer.gradient_clip_val,
    )

    trainer.fit(
        module,
        dataloaders_dict["train_dataloader"],
        dataloaders_dict["val_dataloader"],
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    set_seed(config.random_seed)
    train(config)


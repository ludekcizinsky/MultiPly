from multiply_model import MultiplyModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import glob
from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The `srun` command is available on your system but is not used\."
)


@hydra.main(config_path="confs", config_name="taichi01_base", version_base=None)
def main(opt):
    pl.seed_everything(42)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(opt.output_dir) / "checkpoints",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=100,
        save_top_k=-1,
        save_last=True)
    logger = WandbLogger(
        project=opt.project_name, 
        name=f"{opt.exp}/{opt.run}",
        save_dir=opt.wandb_dir
    )

    trainer = pl.Trainer(
        # gpus=1,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=10000,
        check_val_every_n_epoch=50,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    betas_path = Path(opt.dataset.train.data_dir) / "mean_shape.npy"
    model = MultiplyModel(opt, betas_path)
    trainset = create_dataset(opt.dataset.train)
    validset = create_dataset(opt.dataset.valid)

    if opt.model.is_continue == True:
        # checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        checkpoint = sorted(glob.glob("checkpoints/epoch=*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()
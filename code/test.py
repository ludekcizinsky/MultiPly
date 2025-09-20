from multiply_model import MultiplyModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The `srun` command is available on your system but is not used\."
)


@hydra.main(config_path="confs", config_name="taichi01_base", version_base=None)
def main(opt):
    pl.seed_everything(42)

    ckpt_dir = Path(opt.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(opt.output_dir) / "checkpoints",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=opt.checkpoint_every_n_epochs,
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
        max_epochs=opt.total_num_epochs,
        check_val_every_n_epoch=opt.check_val_every_n_epochs,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    betas_path = Path(opt.dataset.train.data_dir) / "mean_shape.npy"
    model = MultiplyModel(opt, betas_path)
    testset = create_dataset(opt.dataset.test)
    checkpoint = ckpt_dir / "last.ckpt"
    
    trainer.test(model, testset, ckpt_path=checkpoint)

if __name__ == '__main__':
    main()
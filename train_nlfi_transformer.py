from datetime import datetime

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_json import DatasetJSON
from tranformerlm import TransformerLM


@hydra.main(config_path="config", config_name="transformer_config")
def main(cfg: DictConfig):
    seed = 228
    np.random.seed(seed)

    L.seed_everything(seed)

    model = TransformerLM(
        sp_tokenizer_file_name=f"{cfg.tokenizer.dir}/{cfg.tokenizer.model_prefix}.model",
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_hid=cfg.model.d_hid,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.optimizer.learning_rate,
        max_sequence_length=cfg.model.max_sequence_length,
    )

    train_dataset = DatasetJSON(cfg.train.dataset.data, model.tokenizer, model.max_sequence_length)
    eval_dataset = DatasetJSON(cfg.eval.dataset.data, model.tokenizer, model.max_sequence_length)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.dataloader.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.eval.dataloader.batch_size,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    date = datetime.now().strftime("%d.%m.%y_%H.%M")
    version_name = (
        f"{cfg.loggers.tensorboard.subdir}_l_{cfg.optimizer.learning_rate}" f"lr_{cfg.model.d_model}" f"hd_{date}"
    )

    logger = TensorBoardLogger(
        save_dir=cfg.loggers.tensorboard.dir, name=cfg.loggers.tensorboard.subdir, version=version_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.loggers.tensorboard.dir}/{cfg.loggers.tensorboard.subdir}/{version_name}",
        filename="transformerlm_{epoch:2d}_{eval_loss:0.2f}_{eval_pp:0.2f}",
        monitor="eval_loss",
        save_top_k=1,
        verbose=True,
    )

    rich_progress_bar_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description="rgb(197,0,27)",
            progress_bar="rgb(175,0,255)",
            progress_bar_finished="rgb(77,167,73)",
            time="rgb(175,0,255)",
            metrics="rgb(175,175,255)",
        )
    )
    learning_rate_monitor_callback = LearningRateMonitor()

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epoch,
        logger=logger,
        callbacks=[rich_progress_bar_callback, checkpoint_callback, learning_rate_monitor_callback],
        # callbacks=[checkpoint_callback, learning_rate_monitor_callback],
        check_val_every_n_epoch=2,
        accelerator='gpu',
        precision='16',
    )

    trainer.fit(
        model,
        train_dataloader,
        eval_dataloader,
    )

    model.load_from_checkpoint(checkpoint_callback.best_model_path)

    model.eval().cuda()
    print(model.device)
    sample = torch.LongTensor(train_dataset[1][0])
    for i in [0, 10, 20, sample.shape[0] - 1]:
        input_sample = model.tokenizer.decode_ids(sample.tolist()[:i])
        sample_generate_log = (
            f"Original: {model.tokenizer.decode_ids(sample.tolist())}\n"
            f"Input: {input_sample}\n"
            f"Output: {model.generate_sequence(input_sample, 64)}"
        )

        print(sample_generate_log)
        sample_generate_log_file_name = (
            f"{cfg.loggers.tensorboard.dir}/"
            f"{cfg.loggers.tensorboard.subdir}/{version_name}/best_sample_generate_log.txt"
        )

        sample_generate_log_file_mode = "w"

        with open(sample_generate_log_file_name, sample_generate_log_file_mode, encoding="UTF-8") as file:
            file.write(sample_generate_log)


if __name__ == '__main__':
    main()

import random
from datetime import datetime

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_nlfi import DatasetNlfi
from grulm import GRULM
from utils import collate_fn_padding_offseted_targets

# SP_ARTIFACTS_DIR_KEY = 'tokenizer.dir'
# SP_MODEL_PREFIX_KEY = 'tokenizer.model_prefix'
# TRAIN_DATA_FILE_KEY = 'train.dataset.data'
# EVAL_DATA_FILE_KEY = 'eval.dataset.data'
# TB_LOG_DIR_KEY = 'loggers.tensorboard.dir'
# TB_LOG_SUBDIR_KEY = 'loggers.tensorboard.subdir'
# TRAIN_BATCH_SIZE_KEY = 'train.dataloader.batch_size'
# EVAL_BATCH_SIZE_KEY = 'eval.dataloader.batch_size'
# TRAIN_N_LINES_KEY = 'train.dataset.n_lines'
# EVAL_N_LINES_KEY = 'eval.dataset.n_lines'
# LEARNING_RATE_KEY = 'trainer.learning_rate'
# HIDDEN_DIM_KEY = 'model.hidden_dim'
# MAX_EPOCH_KEY = 'trainer.max_epoch'

# sp_artifacts_dir, sp_model_prefix, train_data_file, eval_data_file, log_dir, log_subdir, batch_size, n_lines, \
#         learning_rate, hidden_dim, max_epoch = cfg.train_params.values()


@hydra.main(config_path="config", config_name="base_config")
def train_nlfi(cfg: DictConfig):
    seed = 228
    np.random.seed(seed)

    L.seed_everything(seed)

    grulm = GRULM(
        cfg.model.hidden_dim, f"{cfg.tokenizer.dir}/{cfg.tokenizer.model_prefix}.model", cfg.trainer.learning_rate
    )
    train_dataset = DatasetNlfi(cfg.train.dataset.data, grulm.tokenizer, cfg.train.dataset.n_lines)
    eval_dataset = DatasetNlfi(cfg.eval.dataset.data, grulm.tokenizer, cfg.eval.dataset.n_lines)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.dataloader.batch_size,
        collate_fn=lambda x: collate_fn_padding_offseted_targets(x, grulm.tokenizer.pad_id()),
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.eval.dataloader.batch_size,
        collate_fn=lambda x: collate_fn_padding_offseted_targets(x, grulm.tokenizer.pad_id()),
    )
    date = datetime.now().strftime("%d.%m.%y_%H.%M")
    version_name = (
        f"{cfg.loggers.tensorboard.subdir}_{cfg.train.dataset.n_lines}l_{cfg.trainer.learning_rate}"
        f"lr_{cfg.model.hidden_dim}"
        f"hd_{date}"
    )

    logger = TensorBoardLogger(
        save_dir=cfg.loggers.tensorboard.dir, name=cfg.loggers.tensorboard.subdir, log_graph=True, version=version_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.loggers.tensorboard.dir}/{cfg.loggers.tensorboard.subdir}/{version_name}",
        filename="grulm_{epoch:2d}_{eval_loss:0.2f}_{eval_pp:0.2f}",
        monitor="eval_loss",
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
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epoch, logger=logger, callbacks=[rich_progress_bar_callback, checkpoint_callback]
    )

    trainer.fit(grulm, train_dataloader, eval_dataloader)
    # Text generation based on sequence
    # Loading best checkpoint
    # map_location = {'cuda:0': 'cpu'}
    grulm = GRULM.load_from_checkpoint(
        checkpoint_callback.best_model_path
        # map_location=map_location
    )
    sample = torch.LongTensor(train_dataset[random.randint(2, len(eval_dataloader))][0])
    input_sample = grulm.tokenizer.decode_ids(sample.tolist()[:10])
    sample_generate_log = (
        f"Original: {grulm.tokenizer.decode_ids(sample.tolist())}\n"
        f"Input: {input_sample}\n"
        f"Output: {grulm.generate_sequence(input_sample, 512)}"
    )

    print(sample_generate_log)
    sample_generate_log_file_name = (
        f"{cfg.loggers.tensorboard.dir}/"
        f"{cfg.loggers.tensorboard.subdir}/{version_name}/best_sample_generate_log.txt"
    )

    sample_generate_log_file_mode = "w"

    with open(sample_generate_log_file_name, sample_generate_log_file_mode, encoding="UTF-8") as file:
        file.write(sample_generate_log)


if __name__ == "__main__":
    train_nlfi()  # noqa pylint: disable=no-value-for-parameter

# train.py —— 单验证集 & 验证逐步日志与训练格式一致
from __future__ import annotations
import datetime
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Tuple, cast, List, Dict

import hydra
from datetime import timedelta
import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from jaxtyping import Bool, Float, Integer
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .__init__ import console
from .constants import ANNOTATED_COLUMN, ANNOTATION_ERROR
from .inference import Decoder, GreedyDecoder, ScoredSequence
from .transformer.dataset import collate_batch, remove_modifications
from .transformer.model import InstaNovo
from .types import (
    Peptide,
    PeptideMask,
    PrecursorFeatures,
    ResidueLogits,
    Spectrum,
    SpectrumMask,
)
from .utils import Metrics, ResidueSet, SpectrumDataFrame
from .utils.colorlogging import ColorLog
from .utils.device_handler import check_device
from .utils.s3 import PLCheckpointWrapper, S3FileHandler
from .utils.spectrum_iterable_dataset import SpectrumIterableDataset
from lightning.pytorch.profilers import PyTorchProfiler

load_dotenv()
logger = ColorLog(console, __name__).logger
CONFIG_PATH = Path(__file__).parent.parent / "configs"
warnings.filterwarnings("ignore", message=".*does not have many workers*")


# ----------------- DDP helpers -----------------
def _ddp_rank_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    return rank, world_size

def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def _wait_for_file(path: Path, check_interval: float = 2.0, timeout: float = 36000.0):
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {path} to appear.")
        time.sleep(check_interval)


# ====================== PTL Module ======================
class PTModule(L.LightningModule):
    """Lightning wrapper for model with a single validation dataloader."""

    def __init__(
        self,
        config: DictConfig | dict[str, Any],
        model: InstaNovo,
        decoder: Decoder,
        metrics: Metrics,
        sw: SummaryWriter,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        enable_compile: bool = True,
        fp16: bool = True,
        updates_per_epoch: int = 0,           # 训练步估算
        valid_updates_per_epoch: int = 0,     # 验证步估算（用于日志 ETA/Batch 进度）
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.decoder = decoder
        self.metrics = metrics
        self.sw = sw
        self.optim = optim
        self.scheduler = scheduler

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # ---- 训练期状态 ----
        self.running_loss: float | None = None
        self._last_epoch_train_loss = 0.0
        self.steps = 0  # 全局训练步
        self.train_epoch_start_time: float | None = None
        self.train_start_time: float | None = None
        self.train_epoch_step: int = 0

        # ---- 验证期状态（对齐训练端） ----
        self.valid_running_loss: float | None = None
        self.val_global_steps: int = 0  # 全局验证步
        self.valid_epoch_start_time: float | None = None
        self.valid_epoch_step: int = 0

        # 日志频率缩放（默认以 bs=32 为基准）
        self.step_scale = 32 / config["train_batch_size"]

        # epoch 内剩余时间估算
        self.updates_per_epoch = max(int(updates_per_epoch), 1)
        self.valid_updates_per_epoch = max(int(valid_updates_per_epoch), 1)

        # 验证缓存
        self._reset_val_store()

        # 仅 global_zero 打日志/写 TB
        def _compiled_forward(
            spectra: Tensor,
            precursors: Tensor,
            peptides: Tensor,
            spectra_mask: Tensor,
            peptides_mask: Tensor,
        ) -> Tensor:
            return self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)

        self.compiled_forward = torch.compile(
            _compiled_forward,
            dynamic=False,
            mode="reduce-overhead",
            disable=not enable_compile,
        )

    # 内部：是否允许输出（全局零号 / 非分布式）
    def _log_ok(self) -> bool:
        return (not dist.is_available() or not dist.is_initialized()
                or getattr(self.trainer, "is_global_zero", True))

    def forward(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        peptides: list[str] | Integer[Peptide, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
        peptides_mask: Bool[PeptideMask, " batch"],
    ) -> Float[ResidueLogits, " batch token+1"]:
        return self.model(spectra, precursors, peptides, spectra_mask, peptides_mask)  # type: ignore

    def training_step(
        self,
        batch: tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
    ) -> Float[Tensor, " batch"]:
        if self.train_epoch_start_time is None:
            self.train_epoch_start_time = time.time()
            self.train_epoch_step = 0
        if self.train_start_time is None:
            self.train_start_time = time.time()

        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        peptides_mask = peptides_mask.to(self.device)
        peptides = peptides.to(self.device)

        preds = self.compiled_forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        if self._log_ok() and (
            (self.steps + 1) % max(1, int(self.config.get("console_logging_steps", 2000) * self.step_scale))
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            delta = time.time() - self.train_epoch_start_time
            current_step = self.train_epoch_step + 1
            est_total = delta / max(current_step, 1) * max(self.updates_per_epoch - current_step, 1)
            logger.info(
                f"[TRAIN] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs - 1:02d}"
                f" Step {self.steps + 1:06d}] "
                f"[Batch {current_step:05d}/{self.updates_per_epoch:05d}] "
                f"[{_format_time(delta)}/{_format_time(est_total)}, {(delta / max(current_step,1)):.3f}s/it]: "
                f"train_loss_raw={loss.item():.4f}, running_loss={self.running_loss:.4f}, LR={lr:.6f}"
            )

        if self._log_ok() and (
            (self.steps + 1) % max(1, int(self.config.get("tensorboard_logging_steps", 500) * self.step_scale))
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps + 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps + 1)
            self.sw.add_scalar("optim/lr", lr, self.steps + 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps + 1)

        self.steps += 1
        self.train_epoch_step += 1
        return loss

    # --------- Validation (single dataloader) ----------
    def _reset_val_store(self):
        self.val_preds: List[str] = []
        self.val_targs: List[str] = []
        self.val_losses: List[float] = []

    def on_validation_epoch_start(self) -> None:
        self._reset_val_store()
        self.valid_epoch_start_time = time.time()
        self.valid_epoch_step = 0
        if self._log_ok():
            logger.info(f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}] Start")

    def validation_step(
        self,
        batch: Tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
        batch_idx: int,
    ) -> float:
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        peptides_mask = peptides_mask.to(self.device)
        peptides = peptides.to(self.device)

        with torch.no_grad():
            preds = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        with torch.no_grad():
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config["n_beams"],
                max_length=self.config["max_length"],
            )
            p = cast(List[ScoredSequence], p)

        y = [x.sequence if isinstance(x, ScoredSequence) else "" for x in p]
        targets = list(self.model.batch_idx_to_aa(peptides, reverse=True))

        # 累计
        self.val_preds += y
        self.val_targs += targets
        self.val_losses.append(loss.item())

        # running loss（验证）
        if self.valid_running_loss is None:
            self.valid_running_loss = loss.item()
        else:
            self.valid_running_loss = 0.99 * self.valid_running_loss + (1 - 0.99) * loss.item()

        # 与 train 相同的节流与输出格式
        if self._log_ok() and (
            (self.val_global_steps + 1) % max(1, int(self.config.get("console_logging_steps", 2000) * self.step_scale))
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            delta = time.time() - (self.valid_epoch_start_time or time.time())
            current_step = self.valid_epoch_step + 1
            est_total = delta / max(current_step, 1) * max(self.valid_updates_per_epoch - current_step, 1)
            logger.info(
                f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs - 1:02d}"
                f" Step {self.val_global_steps + 1:06d}] "
                f"[Batch {current_step:05d}/{self.valid_updates_per_epoch:05d}] "
                f"[{_format_time(delta)}/{_format_time(est_total)}, {(delta / max(current_step,1)):.3f}s/it]: "
                f"valid_loss_raw={loss.item():.4f}, running_loss={self.valid_running_loss:.4f}, LR={lr:.6f}"
            )

        # 步计数
        self.val_global_steps += 1
        self.valid_epoch_step += 1

        return float(loss.item())

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch
        if len(self.val_losses) == 0:
            if self._log_ok():
                logger.warning(f"[VALIDATION] [Epoch {epoch:02d}] no batches.")
            return

        valid_loss = float(np.mean(self.val_losses))
        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(self.val_targs, self.val_preds)
        aa_er = self.metrics.compute_aa_er(self.val_targs, self.val_preds)

        if self._log_ok():
            self.sw.add_scalar("eval/valid_loss", valid_loss, epoch)
            self.sw.add_scalar("eval/aa_er", aa_er, epoch)
            self.sw.add_scalar("eval/aa_prec", aa_prec, epoch)
            self.sw.add_scalar("eval/aa_recall", aa_recall, epoch)
            self.sw.add_scalar("eval/pep_recall", pep_recall, epoch)

            logger.info(
                f"[VALIDATION] [Epoch {epoch:02d}] "
                f"valid_loss={valid_loss:.5f}, aa_er={aa_er:.3f}, "
                f"aa_prec={aa_prec:.3f}, aa_recall={aa_recall:.3f}, pep_recall={pep_recall:.3f} "
                f"(n={len(self.val_preds)})"
            )

            logger.info(
                f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] "
                f"train_loss={float(self.running_loss or 0.0):.5f}"
            )

        # 重置本 epoch 的验证态
        self.valid_running_loss = None
        self.valid_epoch_start_time = None
        self.valid_epoch_step = 0
        self._reset_val_store()

    def on_train_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch
        if self.train_start_time is None:
            if self._log_ok():
                logger.warning(f"[TRAIN] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] "
                               f"No training steps executed. Skipping epoch-end logging.")
            self._last_epoch_train_loss = float(self.running_loss) if self.running_loss is not None else 0.0
        else:
            if self._log_ok() and self.running_loss is not None:
                self.sw.add_scalar("train/epoch_loss", self.running_loss, epoch)
                try:
                    self.sw.flush()
                except Exception:
                    pass

            delta = time.time() - cast(float, self.train_start_time)
            est_total = delta / (epoch + 1) * (self.trainer.max_epochs - epoch - 1)
            if self._log_ok():
                logger.info(
                    f"[TRAIN] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] "
                    f"Epoch complete, total time {_format_time(delta)}, remaining "
                    f"{_format_time(est_total)}, {_format_time(delta / (epoch + 1))} per epoch"
                )
            self._last_epoch_train_loss = float(self.running_loss) if self.running_loss is not None else 0.0

        self.running_loss = None
        self.train_epoch_start_time = None
        self.train_epoch_step = 0

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.config = checkpoint["config"]

    def configure_optimizers(self):
        return {
            "optimizer": self.optim,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"},
        }


# ====================== 训练入口 ======================
def train(config: DictConfig) -> None:
    torch.manual_seed(config.get("seed", 101))
    torch.set_float32_matmul_precision("high")

    time_now = datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")

    if S3FileHandler.register_tb():
        config["tb_summarywriter"] = os.environ["AICHOR_LOGS_PATH"]
    else:
        base_log_dir = config["tb_summarywriter"]
        run_folder_name = config.get("run_name", "run") + time_now
        final_log_path = os.path.join(base_log_dir, run_folder_name)
        config["tb_summarywriter"] = final_log_path

    s3 = S3FileHandler()
    training_logger = L.pytorch.loggers.TensorBoardLogger(
        name=config.get("run_name", "no_run_name_specified") + time_now,
        save_dir=config["tb_summarywriter"],
    )
    sw = SummaryWriter(training_logger.log_dir)

    logger.info("Starting transformer training")

    # Vocab / residue set
    residue_set = ResidueSet(
        residue_masses=config["residues"],
        residue_remapping=config.get("residue_remapping", None),
    )
    logger.info(f"Vocab: {residue_set.index_to_residue}")

    logger.info("Loading data")

    # ---------- 加载 SDF ----------
    try:
        train_sdf = SpectrumDataFrame.load(
            source=config.get("train_path"),
            source_type=config.get("source_type", "default"),
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=True,
            partition=config.get("train_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
            preshuffle_across_shards=config.get("preshuffle_shards", False),
            force_convert_to_native=config.get("force_convert_to_native", False),
            verbose=config.get("verbose_loading", True),
        )

        valid_path = config.get("valid_path", None)

        # 单一验证集（若未提供则回退到 train_path 并基于 train_sdf 划分）
        _valid_path = valid_path if valid_path is not None else config.get("train_path")
        valid_sdf = SpectrumDataFrame.load(
            _valid_path,
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=False,
            partition=config.get("valid_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
            force_convert_to_native=config.get("force_convert_to_native", False),
        )

    except ValueError as e:
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, are you trying to run de novo "
                "prediction? Add the --denovo flag"
            ) from e
        else:
            raise

    # 早期子采样
    _tfrac = float(config.get("train_subset", 1.0))
    _vfrac = float(config.get("valid_subset", 1.0))
    if 0.0 < _tfrac < 1.0:
        logger.info(f"[EARLY SUBSET] Train subset fraction={_tfrac}")
        train_sdf.sample_subset(fraction=_tfrac, seed=42)
    if 0.0 < _vfrac < 1.0:
        logger.info(f"[EARLY SUBSET] Valid subset fraction={_vfrac}")
        valid_sdf.sample_subset(fraction=_vfrac, seed=42)

    # 若未指定 valid_path，则从训练集拆分（保持向后兼容）
    if config.get("valid_path", None) is None:
        logger.info("Validation path not specified, generating from training set.")
        sequences = list(train_sdf.get_unique_sequences())
        sequences = sorted({remove_modifications(x) for x in sequences})
        train_unique, valid_unique = train_test_split(
            sequences,
            test_size=config.get("valid_subset_of_train"),
            random_state=42,
        )
        train_unique = set(train_unique)
        valid_unique = set(valid_unique)
        train_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in train_unique)
        valid_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in valid_unique)

        split_path = os.path.join(config.get("model_save_folder_path", "./checkpoints"), "splits.csv")
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        pd.DataFrame(
            {
                "unmodified_sequence": list(train_unique) + list(valid_unique),
                "split": ["train"] * len(train_unique) + ["valid"] * len(valid_unique),
            }
        ).to_csv(str(split_path), index=False)
        logger.info(f"Data splits saved to {split_path}")

    # 数据完整性检查（联合 train + valid）
    if config.get("perform_data_checks", True):
        logger.info(f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows.")
        residue_set_supported = set(residue_set.vocab)
        residue_set_supported.update(set(residue_set.residue_remapping.keys()))
        data_residues = set()
        data_residues.update(train_sdf.get_vocabulary(residue_set.tokenize))
        data_residues.update(valid_sdf.get_vocabulary(residue_set.tokenize))

        if len(data_residues - residue_set_supported) > 0:
            logger.warning("Unsupported residues found in evaluation set! These rows will be dropped.")
            logger.info(f"New residues found: \n{data_residues - residue_set_supported}")
            logger.info(f"Residues supported: \n{residue_set_supported}")

            orig_train, orig_valid = len(train_sdf), len(valid_sdf)
            train_sdf.filter_rows(
                lambda row: all(res in residue_set_supported for res in set(residue_set.tokenize(row[ANNOTATED_COLUMN])))
            )
            valid_sdf.filter_rows(
                lambda row: all(res in residue_set_supported for res in set(residue_set.tokenize(row[ANNOTATED_COLUMN])))
            )
            logger.warning(
                f"{orig_train - len(train_sdf):,d} "
                f"({(orig_train - len(train_sdf)) / max(orig_train,1) * 100:.2f}%) training rows dropped."
            )
            logger.warning(
                f"[valid] dropped {orig_valid - len(valid_sdf):,d} "
                f"({(orig_valid - len(valid_sdf)) / max(orig_valid,1) * 100:.2f}%) validation rows."
            )

        # 过滤不合理电荷
        orig_train, orig_valid = len(train_sdf), len(valid_sdf)
        train_sdf.filter_rows(lambda row: (row["precursor_charge"] <= config.get("max_charge", 10)) and (row["precursor_charge"] > 0))
        if len(train_sdf) < orig_train:
            logger.warning(
                f"Found {orig_train - len(train_sdf)} rows in training set with invalid charge; removed."
            )
        valid_sdf.filter_rows(lambda row: (row["precursor_charge"] <= config.get("max_charge", 10)) and (row["precursor_charge"] > 0))
        if len(valid_sdf) < orig_valid:
            logger.warning(
                f"Found {orig_valid - len(valid_sdf)} rows in validation set with invalid charge; removed."
            )

    # ---------- 将过滤后的 SDF 保存为分片 ----------
    tmp_root = os.path.join(config.get("model_save_folder_path", "./checkpoints"),
                            "_iter_shards", config.get("run_name", "run"))
    train_tmp_dir = Path(tmp_root) / "train"
    valid_tmp_dir = Path(tmp_root) / "valid"
    train_tmp_dir.mkdir(parents=True, exist_ok=True)
    valid_tmp_dir.mkdir(parents=True, exist_ok=True)

    train_partition = config.get("train_partition", "train")
    valid_partition = config.get("valid_partition", "valid")
    dataset_name = config.get("dataset_name", "ms")
    overwrite = bool(config.get("overwrite_iter_shards", False))

    def _has_shards(d: Path) -> bool:
        return any(d.glob("*.parquet")) or any(d.glob("*.ipc"))

    # --- TRAIN shards ---
    train_done = train_tmp_dir / ".shards_done"
    if _is_rank_zero():
        if overwrite or not _has_shards(train_tmp_dir):
            logger.info(f"Saving filtered TRAIN shards to: {train_tmp_dir}")
            train_done.exists() and train_done.unlink(missing_ok=True)
            train_sdf.save(
                train_tmp_dir,
                partition=train_partition,
                name=dataset_name,
                max_shard_size=config.get("max_shard_size", 100_000),
            )
            train_done.write_text("ok")
        else:
            logger.info(f"Found existing TRAIN shards in {train_tmp_dir}, skipping save.")
            train_done.exists() or train_done.write_text("ok")
    else:
        _wait_for_file(train_done)

    # --- VALID shards (single) ---
    valid_done = valid_tmp_dir / ".shards_done"
    if _is_rank_zero():
        if overwrite or not _has_shards(valid_tmp_dir):
            logger.info(f"Saving filtered VALID shards to: {valid_tmp_dir}")
            valid_done.exists() and valid_done.unlink(missing_ok=True)
            valid_sdf.save(
                valid_tmp_dir,
                partition=valid_partition,
                name=dataset_name,
                max_shard_size=config.get("max_shard_size", 100_000),
            )
            valid_done.write_text("ok")
        else:
            logger.info(f"Found existing VALID shards in {valid_tmp_dir}, skipping save.")
            valid_done.exists() or valid_done.write_text("ok")
    else:
        _wait_for_file(valid_done)

    # 估算 world_size（用于训练/验证步数日志与 limit_train_batches）
    _, _ws_env = _ddp_rank_world_size()
    _ws_guess = _ws_env if _ws_env > 1 else (torch.cuda.device_count() or 1)

    updates_per_epoch = int(np.ceil(len(train_sdf) / (config["train_batch_size"] * _ws_guess)))
    valid_updates_per_epoch = int(np.ceil(len(valid_sdf) / (config["predict_batch_size"] * _ws_guess)))

    logger.info(
        f"Data loaded: {len(train_sdf):,} training samples; {len(valid_sdf):,} validation samples"
    )

    # warmup 长度检查
    if config.get("warmup_iters", 100_000) > max(updates_per_epoch, 1):
        logger.warning("Model warmup is greater than one epoch of the training set. Check warmup_iters in config")

    # checkpoint 频率估算
    if config.get("save_model", True):
        total_epochs = config.get("epochs", 30)
        epochs_per_save = config["ckpt_interval"] / max(updates_per_epoch, 1)
        if epochs_per_save > total_epochs:
            logger.warning(
                f"Model checkpoint will never save. Attempting to save every {epochs_per_save:.2f} "
                f"epochs but only training for {total_epochs:d} epochs. Check ckpt_interval in config."
            )
        else:
            logger.info(f"Model checkpointing every {epochs_per_save:.2f} epochs.")
    else:
        logger.info("Model saving disabled")

    # 黑名单检查
    if config.get("blacklist", None):
        logger.info("Checking if any training set overlaps with blacklisted sequences...")
        blacklist_df = pd.read_csv(config["blacklist"])
        train_sequences = pl.Series(list(train_sdf.get_unique_sequences()))
        leakage = any(
            train_sequences.map_elements(remove_modifications, return_dtype=pl.String).is_in(
                blacklist_df["sequence"]
            )
        )
        if leakage:
            raise ValueError("Portion of training set sequences overlaps with blacklisted sequences.")
        else:
            logger.info("No blacklisted sequences!")

    # 训练/验证集泄露检查
    if config.get("perform_data_checks", True):
        logger.info("Checking if validation set overlaps with training set...")
        train_sequences = pl.Series(list(train_sdf.get_unique_sequences()))
        valid_sequences = pl.Series(list(valid_sdf.get_unique_sequences()))
        leakage = any(valid_sequences.is_in(train_sequences))
        if leakage:
            raise ValueError("Portion of validation set sequences overlaps with training set.")
        logger.info("No data leakage!")

    # ---------- IterableDataset + DataLoader ----------
    pin_cuda = torch.cuda.is_available()
    _tw = int(config.get("num_workers", 4))
    _vw = int(config.get("eval_num_workers", 4))

    train_ds = SpectrumIterableDataset(
        data_path=train_tmp_dir,
        residue_set=residue_set,
        split=train_partition,
        n_peaks=config["n_peaks"],
        peptide_pad_length=(config.get("max_length", 40) if config.get("compile_model", False) else 0),
        pad_spectrum_max_length=(config.get("compile_model", False) or config.get("use_flash_attention", False)),
        bin_spectra=config.get("conv_peak_encoder", False),
        batch_rows=config.get("arrow_batch_rows", 65536),
        shuffle_buffer=config.get("shuffle_buffer", 0),
        use_threads=False,
        use_sus_preprocess=config.get("use_sus_preprocess", True),
        shard_across_ranks=True,
        shuffle_files=True,
    )
    _train_kwargs = dict(
        batch_size=config["train_batch_size"],
        num_workers=_tw,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=pin_cuda,
        persistent_workers=(_tw > 0),
        drop_last=True,
        multiprocessing_context="spawn",
        **({"pin_memory_device": "cuda"} if pin_cuda else {}),
    )
    if _tw > 0:
        _train_kwargs["prefetch_factor"] = config.get("prefetch_factor", 1)
    train_dl = DataLoader(train_ds, **_train_kwargs)

    # 单验证集 DataLoader
    _valid_kwargs = dict(
        batch_size=config["predict_batch_size"],
        num_workers=_vw,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=pin_cuda,
        persistent_workers=(_vw > 0),
        drop_last=False,
        multiprocessing_context="spawn",
        **({"pin_memory_device": "cuda"} if pin_cuda else {}),
    )
    if _vw > 0:
        _valid_kwargs["prefetch_factor"] = config.get("eval_prefetch_factor", 1)

    valid_ds = SpectrumIterableDataset(
        data_path=valid_tmp_dir,
        residue_set=residue_set,
        split=valid_partition,
        n_peaks=config["n_peaks"],
        pad_spectrum_max_length=(config.get("compile_model", False) or config.get("use_flash_attention", False)),
        bin_spectra=config.get("conv_peak_encoder", False),
        batch_rows=config.get("arrow_batch_rows", 65536),
        shuffle_buffer=0,
        use_threads=False,
        use_sus_preprocess=config.get("use_sus_preprocess", True),
        shard_across_ranks=True,   # DDP 下验证也分片
        shuffle_files=False,       # 验证不打乱
    )
    valid_dl = DataLoader(valid_ds, **_valid_kwargs)

    # 日志
    step_scale = 32 / config["train_batch_size"]
    if _is_rank_zero():
        logger.info(f"Updates per epoch: {updates_per_epoch:,}, valid updates per epoch: {valid_updates_per_epoch:,}, step_scale={step_scale}")

    # ---------- 初始化模型 ----------
    model = InstaNovo(
        residue_set=residue_set,
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_charge=config["max_charge"],
        use_flash_attention=config["use_flash_attention"],
        conv_peak_encoder=config["conv_peak_encoder"],
    )

    # checkpoint 恢复
    resume_checkpoint_path = None if config["train_from_scratch"] else config["resume_checkpoint"]
    if resume_checkpoint_path is not None:
        logger.info(f"Loading model checkpoint from '{resume_checkpoint_path}'")
        model_state = torch.load(resume_checkpoint_path, map_location="cpu")
        if "state_dict" in model_state:
            model_state = {k.replace("model.", ""): v for k, v in model_state["state_dict"].items()}
        aa_embed_size = model_state["head.weight"].shape[0]
        if aa_embed_size != len(residue_set):
            state_keys = ["head.weight", "head.bias", "aa_embed.weight"]
            logger.warning(f"Model expects vocab size of {len(residue_set)}, checkpoint has {aa_embed_size}.")
            logger.warning("Assuming a change was made to the residues in the configuration file.")
            logger.warning(f"Automatically converting {state_keys} to match expected.")
            new_model_state = model.state_dict()
            resolution = config.get("residue_conflict_resolution", "delete")
            for k in state_keys:
                tmp = torch.normal(
                    mean=0, std=1.0 / np.sqrt(config["dim_model"]),
                    size=new_model_state[k].shape, dtype=new_model_state[k].dtype,
                )
                if "bias" in k:
                    tmp = torch.zeros_like(tmp)
                if resolution == "delete":
                    del model_state[k]
                elif resolution == "random":
                    model_state[k] = tmp
                elif resolution == "partial":
                    tmp[:aa_embed_size] = model_state[k][: min(tmp.shape[0], aa_embed_size)]
                    model_state[k] = tmp
                else:
                    raise ValueError(f"Unknown residue_conflict_resolution type '{resolution}'")
            logger.warning(f"Model checkpoint has {len(state_keys)} weights updated with '{resolution}' conflict resolution")
        k_missing = np.sum([x not in list(model_state.keys()) for x in list(model.state_dict().keys())])
        if k_missing > 0:
            logger.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing = np.sum([x not in list(model.state_dict().keys()) for x in list(model_state.keys())])
        if k_missing > 0:
            logger.warning(f"Model state is missing {k_missing} keys!")
        model.load_state_dict(model_state, strict=False)

    logger.info(f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters")

    # 设备
    device = check_device(config=config)
    model = model.to(device)
    if config.get("fp16", True) and device.lower() == "cpu":
        logger.warning("fp16 is enabled but device type is cpu. fp16 will be disabled.")
        config["fp16"] = False

    decoder = GreedyDecoder(model=model)
    metrics = Metrics(residue_set, config["isotope_error_range"])

    # 可选前体质量 sanity check
    if config.get("validate_precursor_mass", True):
        logger.info("Sanity checking precursor masses for training set...")
        train_sdf.validate_precursor_mass(metrics)
        logger.info("Sanity checking precursor masses for validation set...")
        valid_sdf.validate_precursor_mass(metrics)

    # 优化器 & warmup
    optim = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    strategy = _get_strategy()

    # profiler（可选）
    if config["profiler"]:
        profiler_log_dir = os.path.join(config["tb_summarywriter"], "profiler")
        logger.info(f"Profiler trace will be saved to: {profiler_log_dir}")
        profiler = PyTorchProfiler(
            dirpath=profiler_log_dir,
            filename="profile_trace",
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler = None

    # checkpoint 回调
    if config["save_model"]:
        logger.info("Model saving enabled")
        s3_ckpt_path = S3FileHandler.convert_to_s3_output(config["model_save_folder_path"])
        if S3FileHandler._aichor_enabled():
            callbacks = [
                PLCheckpointWrapper(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                    s3_ckpt_path=s3_ckpt_path,
                    s3=S3FileHandler(),
                    strategy=strategy,
                )
            ]
        else:
            callbacks = [
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                )
            ]
        logger.info(f"Saving every {config['ckpt_interval']} training steps to {s3_ckpt_path}")
    else:
        logger.info("Model saving disabled")
        callbacks = None

    logger.info("Initializing Pytorch Lightning trainer.")

    # val_check_interval 兼容 float
    _vci_cfg = config.get("val_check_interval", 1.0)
    if isinstance(_vci_cfg, float):
        _vci = 1.0 if _vci_cfg >= 1.0 else max(1, int(np.ceil(updates_per_epoch * _vci_cfg)))
    else:
        _vci = int(_vci_cfg)

    trainer = L.pytorch.Trainer(
        profiler=profiler,
        accelerator="gpu" if "cuda" in device else "cpu",
        precision="16-mixed" if config["fp16"] else None,
        callbacks=callbacks,
        devices="auto",
        logger=training_logger,
        max_epochs=config["epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        accumulate_grad_batches=config["grad_accumulation"],
        gradient_clip_val=config["gradient_clip_val"],
        enable_progress_bar=False,
        strategy=strategy,
        val_check_interval=_vci,
        limit_train_batches=updates_per_epoch,
        limit_val_batches=valid_updates_per_epoch,# IterableDataset 训练步数由我们估算
    )

    logger.info("InstaNovo training started.")
    ptmodel = PTModule(
        config, model, decoder, metrics, sw, optim, scheduler,
        enable_compile=config["compile_model"],
        fp16=config["fp16"],
        updates_per_epoch=updates_per_epoch,
        valid_updates_per_epoch=valid_updates_per_epoch,
    )
    trainer.fit(ptmodel, train_dl, valid_dl)
    logger.info("InstaNovo training finished.")


def _get_strategy() -> DDPStrategy | str:
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True, timeout=timedelta(hours=16))
    return "auto"


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# --- 脚本主入口 ---
@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="instanovo")
def main(config: DictConfig) -> None:
    logger.info("Initializing training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # （如需）设置 Neptune token
    try:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
        author_email, _ = author_email.split("@")
        author_email = author_email.replace("-", "_").replace(".", "_").upper()
        logger.info(f"Checking for Neptune API token under {author_email}__NEPTUNE_API_TOKEN.")
        author_api_token = os.environ.get(f"{author_email}__NEPTUNE_API_TOKEN", None)
        if author_api_token:
            os.environ["NEPTUNE_API_TOKEN"] = author_api_token
            logger.info(f"Set token for {author_email}.")
        else:
            logger.info(f"Neptune credentials for user {author_email} not found.")
    except KeyError:
        logger.debug("Not running on AIchor, skipping Neptune token setup.")

    # 将子配置拍平
    sub_configs_list = ["model", "dataset"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"Imported hydra config:\n{OmegaConf.to_yaml(config)}")
    train(config)


if __name__ == "__main__":
    main()

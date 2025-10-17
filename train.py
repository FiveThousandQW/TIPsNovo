from __future__ import annotations
import os
from torch.utils.data import Dataset
import datetime
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Tuple, cast
from tqdm import tqdm # 确保 import 了 tqdm
import hydra
import lightning as L
import neptune # 实验跟踪工具
import numpy as np
import pandas as pd
import polars as pl
import torch
from dotenv import load_dotenv # 用于从 .env 文件加载环境变量
from jaxtyping import Bool, Float, Integer # 提供更丰富的张量类型提示
from lightning.pytorch.strategies import DDPStrategy
from neptune.integrations.python_logger import NeptuneHandler # Neptune 日志处理器
from neptune.internal.utils.git import GitInfo # Neptune Git 信息工具
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict # Hydra 的配置处理工具
from sklearn.model_selection import train_test_split # 用于划分训练/验证集
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # TensorBoard 日志记录器
from .__init__ import console
from .constants import ANNOTATED_COLUMN, ANNOTATION_ERROR
from .inference import Decoder, GreedyDecoder, ScoredSequence
from .transformer.dataset import SpectrumDataset, collate_batch, remove_modifications
from .transformer.model import InstaNovo  # Transformer 模型主类
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
from lightning.pytorch.profilers import PyTorchProfiler
from torch.utils.data.distributed import DistributedSampler

load_dotenv()

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"

# 忽略 PyTorch Lightning DataLoader 的一些常见警告
warnings.filterwarnings("ignore", message=".*does not have many workers*")


# PTModule 是一个封装了模型训练和验证逻辑的 PyTorch Lightning 模块
class PTModule(L.LightningModule):
    """PTL wrapper for model."""

    def __init__(
        self,
        config: DictConfig | dict[str, Any],
        model: InstaNovo,
        decoder: Decoder,
        metrics: Metrics,
        sw: SummaryWriter,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        disable_compile: bool = False,
        fp16: bool = True,
        validation_groups: pl.Series | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.decoder = decoder
        self.metrics = metrics
        self.sw = sw # SummaryWriter，用于向 TensorBoard 或 Neptune 写入日志
        self.optim = optim # 优化器，如 Adam
        self.scheduler = scheduler  # 学习率调度器
        self.validation_groups = validation_groups # 验证集分组信息
        if validation_groups is not None:
            self.groups = validation_groups.unique()

        # 定义损失函数：交叉熵损失，忽略索引为 0 的 token (通常是 padding)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # 初始化用于跟踪训练过程的状态变量
        self.running_loss = None
        self._reset_valid_metrics() # 重置验证指标
        self.steps = 0 # 全局训练步数
        self.train_epoch_start_time: float | None = None
        self.train_start_time: float | None = None
        self.valid_epoch_start_time: float | None = None
        self.valid_epoch_step = 0

        # Update rates based on bs=32
        # 根据批处理大小调整日志记录频率
        self.step_scale = 32 / config["train_batch_size"]

        # 使用 torch.compile 对前向传播函数进行即时编译（JIT），以大幅提升速度
        @torch.compile(dynamic=False, mode="reduce-overhead", disable=disable_compile)

        # 使用 autocast 自动进行混合精度计算（如果启用 fp16）
        @torch.autocast("cuda", dtype=torch.float16, enabled=fp16)
        def compiled_forward(
            spectra: Tensor,
            precursors: Tensor,
            peptides: Tensor,
            spectra_mask: Tensor,
            peptides_mask: Tensor,
        ) -> Tensor:
            """编译后的前向传播函数。"""
            # 调用下面的 forward 方法
            """Compiled forward pass."""
            return self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)

        self.compiled_forward = compiled_forward


    def forward(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        peptides: list[str] | Integer[Peptide, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
        peptides_mask: Bool[PeptideMask, " batch"],
    ) -> Float[ResidueLogits, " batch token+1"]:
        """Model forward pass."""
        """模型的标准前向传播，直接调用内部的 InstaNovo 模型。"""
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
        """A single training step.

        Args:
            batch (tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]) :
                A batch of MS/MS spectra, precursor information, and peptide
                sequences as torch Tensors.

        Returns:
            torch.FloatTensor: training loss
        """
        """
        定义单个训练步骤的逻辑。PyTorch Lightning 会自动调用此函数。
        """

        # 记录 epoch 和训练开始的时间
        if self.train_epoch_start_time is None:
            self.train_epoch_start_time = time.time()
            self.valid_epoch_start_time = None
        if self.train_start_time is None:
            self.train_start_time = time.time()

        # 从批数据中解包并移动到指定设备（CPU/GPU）
        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)
        peptides_mask = peptides_mask.to(self.device)
        peptides = peptides.to(self.device)

        # 调用（编译过的）前向传播函数，得到模型的预测输出 (logits)
        preds = self.compiled_forward(spectra, precursors, peptides, spectra_mask, peptides_mask)

        # 调整预测和标签的形状以匹配损失函数的要求
        # 切掉最后一个 token 的预测（因为不需要预测 EOS 之后的东西）
        # Cut off EOS's prediction, ignore_index should take care of masking
        # EOS at positions < sequence_length will have a label of ignore_index
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])

        # # 计算损失
        loss = self.loss_fn(preds, peptides.flatten())

        # 更新平滑后的运行损失（Exponential Moving Average）
        if self.running_loss is None:
            self.running_loss = loss.item()
        else:
            self.running_loss = 0.99 * self.running_loss + (1 - 0.99) * loss.item()

        # 定期在控制台打印训练日志
        if (
            (self.steps + 1) % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            delta = time.time() - self.train_epoch_start_time
            epoch_step = self.steps % len(self.trainer.train_dataloader)
            est_total = (
                delta / (epoch_step + 1) * (len(self.trainer.train_dataloader) - epoch_step - 1)
            )

            logger.info(
                f"[TRAIN] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs - 1:02d}"
                f" Step {self.steps + 1:06d}] "
                f"[Batch {epoch_step + 1:05d}/{len(self.trainer.train_dataloader):05d}] "
                f"[{_format_time(delta)}/{_format_time(est_total)}, "
                f"{(delta / (epoch_step + 1)):.3f}s/it]: "
                f"train_loss_raw={loss.item():.4f}, "
                f"running_loss={self.running_loss:.4f}, LR={lr:.6f}"
            )

        # 定期向 TensorBoard/Neptune 写入日志
        if (self.steps + 1) % int(
            self.config.get("tensorboard_logging_steps", 500) * self.step_scale
        ) == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.sw.add_scalar("train/loss_raw", loss.item(), self.steps + 1)
            self.sw.add_scalar("train/loss_smooth", self.running_loss, self.steps + 1)
            self.sw.add_scalar("optim/lr", lr, self.steps + 1)
            self.sw.add_scalar("optim/epoch", self.trainer.current_epoch, self.steps + 1)

        # 更新全局步数
        self.steps += 1

        # 返回当前批次的损失值
        return loss

    def validation_step(
        self,
        batch: Tuple[
            Float[Spectrum, " batch"],
            Float[PrecursorFeatures, " batch"],
            Bool[SpectrumMask, " batch"],
            Integer[Peptide, " batch"],
            Bool[PeptideMask, " batch"],
        ],
        *args: Any,
    ) -> float:
        """Single validation step."""
        """
        定义单个验证步骤的逻辑。
        """
        if self.valid_epoch_start_time is None:
            logger.info(
                f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}/"
                f"{self.trainer.max_epochs - 1:02d}] Starting validation."
            )
            self.valid_epoch_start_time = time.time()

        spectra, precursors, spectra_mask, peptides, peptides_mask = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)
        spectra_mask = spectra_mask.to(self.device)

        # Loss
        peptides = peptides.to(self.device)

        # 在不计算梯度的模式下执行，以加速并节省内存
        with torch.no_grad():
            preds = self.forward(spectra, precursors, peptides, spectra_mask, peptides_mask)
        # Cut off EOS's prediction, ignore_index should take care of masking
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        loss = self.loss_fn(preds, peptides.flatten())

        # Greedy decoding
        with torch.no_grad():
            p = self.decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=self.config["n_beams"],
                max_length=self.config["max_length"],
            )
            p = cast(list[ScoredSequence], p)

        # 将预测和真实标签的索引转换回氨基酸序列
        y = [x.sequence if isinstance(x, ScoredSequence) else [] for x in p]
        targets = list(self.model.batch_idx_to_aa(peptides, reverse=True))

        # 收集当前批次的预测和真实标签
        self.valid_predictions += y
        self.valid_targets += targets

        # 计算当前批次的性能指标
        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targets, y)
        aa_er = self.metrics.compute_aa_er(targets, y)

        # 将指标追加到列表中，以便 epoch 结束后计算平均值
        self.valid_metrics["valid_loss"].append(loss.item())
        self.valid_metrics["aa_er"].append(aa_er)
        self.valid_metrics["aa_prec"].append(aa_prec)
        self.valid_metrics["aa_recall"].append(aa_recall)
        self.valid_metrics["pep_recall"].append(pep_recall)

        # 定期打印验证进度日志
        if (
            (self.valid_epoch_step + 1)
            % int(self.config.get("console_logging_steps", 2000) * self.step_scale)
        ) == 0:
            delta = time.time() - self.valid_epoch_start_time
            epoch_step = self.valid_epoch_step % len(self.trainer.val_dataloaders)
            est_total = (
                delta / (epoch_step + 1) * (len(self.trainer.val_dataloaders) - epoch_step - 1)
            )

            logger.info(
                f"[VALIDATION] [Epoch {self.trainer.current_epoch:02d}/"
                f"{self.trainer.max_epochs - 1:02d} Step {self.steps + 1:06d}] "
                f"[Batch {epoch_step:05d}/{len(self.trainer.val_dataloaders):05d}] "
                f"[{_format_time(delta)}/{_format_time(est_total)}, "
                f"{(delta / (epoch_step + 1)):.3f}s/it]"
            )

        self.valid_epoch_step += 1

        return float(loss.item())

    def on_train_epoch_end(self) -> None:
        """Log the training loss at the end of each epoch."""
        """在每个训练 epoch 结束时调用。"""
        # 记录平滑后的训练损失
        epoch = self.trainer.current_epoch
        self.sw.add_scalar("eval/train_loss", self.running_loss, epoch)

        delta = time.time() - cast(float, self.train_start_time)
        epoch = self.trainer.current_epoch
        est_total = delta / (epoch + 1) * (self.trainer.max_epochs - epoch - 1)
        logger.info(
            f"[TRAIN] [Epoch {self.trainer.current_epoch:02d}/{self.trainer.max_epochs - 1:02d}] "
            f"Epoch complete, total time {_format_time(delta)}, remaining time "
            f"{_format_time(est_total)}, {_format_time(delta / (epoch + 1))} per epoch"
        )

        # 重置状态
        self.running_loss = None
        self.train_epoch_start_time = None
        self.train_epoch_step = 0

    def on_validation_epoch_start(self) -> None:
        """在每个验证 epoch 开始时调用。"""
        # 清空用于收集验证结果的列表
        """Reset validation predictions at the start of the epoch."""
        self.valid_predictions: list[list[str]] = []
        self.valid_targets: list[list[str]] = []

    def on_validation_epoch_end(self) -> None:
        """在每个验证 epoch 结束时调用。"""
        # 计算整个验证集上的平均指标
        """Log the validation metrics at the end of each epoch."""
        epoch = self.trainer.current_epoch
        if self.steps == 0:
            # Don't record sanity check validation
            self._reset_valid_metrics()
            return
        for k, v in self.valid_metrics.items():
            self.sw.add_scalar(f"eval/{k}", np.mean(v), epoch)

        # 打印总结性的日志
        valid_loss = np.mean(self.valid_metrics["valid_loss"])
        logger.info(
            f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] "
            f"train_loss={self.running_loss if self.running_loss else 0:.5f}, "
            f"valid_loss={valid_loss:.5f}"
        )
        logger.info(f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] Metrics:")
        for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
            val = np.mean(self.valid_metrics[metric])
            logger.info(
                f"[VALIDATION] [Epoch {epoch:02d}/{self.trainer.max_epochs - 1:02d}] - "
                f"{metric:11s}{val:.3f}"
            )


        # Validation group logging
        # 如果有分组验证集，则为每个组单独计算和记录指标
        if self.validation_groups is not None:
            preds = pl.Series(self.valid_predictions)
            targs = pl.Series(self.valid_targets)
            df_constructor = pl.DataFrame({'targs':targs, 'preds': preds})
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            output_name = f"{self.config['run_name']}{timestamp}_target_pred.csv"
            out_path = Path(self.config['model_save_folder_path']) / output_name.replace(".csv", ".ndjson")
            df_constructor.write_ndjson(str(out_path))
            # test

            assert len(preds) == len(self.validation_groups)
            assert len(targs) == len(self.validation_groups)

            for group in self.groups:
                idx = self.validation_groups == group
                aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(
                    targs.filter(idx), preds.filter(idx)
                )
                aa_er = self.metrics.compute_aa_er(targs.filter(idx), preds.filter(idx))
                self.sw.add_scalar(f"eval/{group}_aa_er", aa_er, epoch)
                self.sw.add_scalar(f"eval/{group}_aa_prec", aa_prec, epoch)
                self.sw.add_scalar(f"eval/{group}_aa_recall", aa_recall, epoch)
                self.sw.add_scalar(f"eval/{group}_pep_recall", pep_recall, epoch)

        self.valid_predictions = []
        self.valid_targets = []

        # 重置状态和指标
        self.valid_epoch_start_time = None
        self.valid_epoch_step = 0
        self._reset_valid_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save config with checkpoint."""
        """在保存检查点时调用。"""
        # 将配置文件也一并保存到检查点文件中
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attempt to load config with checkpoint."""
        """在加载检查点时调用。"""
        # 从检查点文件中恢复配置
        self.config = checkpoint["config"]

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
        """
        配置优化器和学习率调度器。这是 Lightning 要求必须实现的方法。
        """
        # 返回一个包含优化器和调度器的列表/字典

        """Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns:
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        return [self.optim], {"scheduler": self.scheduler, "interval": "step"}

    def _reset_valid_metrics(self) -> None:
        """一个辅助方法，用于重置存储验证指标的字典。"""
        valid_metrics = ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_recall"]
        self.valid_metrics: dict[str, list[float]] = {x: [] for x in valid_metrics}

def train(
    config: DictConfig,
) -> None:
    """Training function."""
    # 设置随机种子以保证结果可复现
    torch.manual_seed(config.get("seed", 101))
    torch.set_float32_matmul_precision("high") # 调整矩阵乘法精度以提升性能

    # --- 1. 实验跟踪与日志设置 ---
    time_now = datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")

    # ... (设置 TensorBoard 或 Neptune 的日志保存路径) ...
    if S3FileHandler.register_tb():
        config["tb_summarywriter"] = os.environ["AICHOR_LOGS_PATH"]
    else:
        # --- 修改开始 ---
        # 1. 从配置中获取基础日志目录 (例如: .../log)
        base_log_dir = config["tb_summarywriter"]
        # 2. 从配置中获取运行名称并附加上时间戳，作为子文件夹的名称
        run_folder_name = config.get("run_name", "run") + time_now
        # 3. 使用 os.path.join 将它们安全地拼接成一个完整的路径
        final_log_path = os.path.join(base_log_dir, run_folder_name)
        # 4. 更新配置中的路径为新生成的完整路径
        config["tb_summarywriter"] = final_log_path
        # --- 修改结束 ---

    s3 = S3FileHandler()

    # 默认使用 TensorBoard
    sw = SummaryWriter(config["tb_summarywriter"])
    training_logger = L.pytorch.loggers.TensorBoardLogger(
        name=config.get("run_name", "no_run_name_specified") + time_now,
        save_dir=config["tb_summarywriter"],
    )

    logger.info("Starting transformer training")

    # --- 2. 数据准备与加载 ---

    # 初始化氨基酸词汇表
    # Transformer vocabulary
    residue_set = ResidueSet(
        residue_masses=config["residues"],
        residue_remapping=config.get("residue_remapping", None),
    )
    logger.info(f"Vocab: {residue_set.index_to_residue}")

    logger.info("Loading data")

    validation_group_mapping = None # 初始化验证集分组映射

    ## 使用直接完成准备的，序列化完毕的data
    # logger.info("Using ready data")
    # if config['use_readyData']:
    #     train_cache = torch.load(config['train_readyData'])
    #     valid_cache = torch.load(config['test_readyData'])
    # else:
    # 使用 SpectrumDataFrame 加载训练和验证数据集
    try:
        # 使用自定义的 SpectrumDataFrame 类加载训练数据。这个类非常强大，支持懒加载、分片、预打乱等功能
        train_sdf = SpectrumDataFrame.load(
            source=config.get("train_path"),
            source_type=config.get("source_type", "default"),
            lazy=config.get("lazy_loading", False),  # 是否懒加载（处理大数据时很有用）
            is_annotated=True,
            shuffle=True,
            partition=config.get("train_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
            preshuffle_across_shards=config.get("preshuffle_shards", False),
            force_convert_to_native=config.get("force_convert_to_native", False),
            verbose=config.get("verbose_loading", True),
        )

        # 加载验证数据
        valid_path = config.get("valid_path", None)
        if valid_path is not None:
            # 如果验证路径是一个字典，则表示有多个命名的验证组
            if OmegaConf.is_dict(valid_path):
                logger.info("Found grouped validation datasets.")
                validation_group_mapping = _get_filepath_mapping(valid_path)
                _valid_path = list(valid_path.values())
            else:
                _valid_path = valid_path
        else:
            # 如果没有指定验证路径，则默认使用训练路径（后续会从中划分
            _valid_path = config.get("train_path")

        # 加载验证数据
        valid_sdf = SpectrumDataFrame.load(
            _valid_path,
            lazy=config.get("lazy_loading", False),
            is_annotated=True,
            shuffle=False,
            partition=config.get("valid_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
            force_convert_to_native=config.get("force_convert_to_native", False),
            add_source_file_column=True,  # used to track validation groups # 添加来源文件列，用于分组验证
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        # 如果加载时发现注释列缺失，则抛出更友好的错误信息
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, are you trying to run de novo "
                "prediction? Add the --denovo flag"
            ) from e
        else:
            raise

    # 如果没有指定验证集，则从训练集中自动划分一部分
    if config.get("valid_path", None) is None:
        logger.info("Validation path not specified, generating from training set.")
        # 获取所有唯一的肽段序列（不含修饰）
        sequences = list(train_sdf.get_unique_sequences())
        sequences = sorted({remove_modifications(x) for x in sequences})
        # 使用 scikit-learn 的 train_test_split 进行划分，确保同一肽段不会同时出现在训练集和验证集中
        train_unique, valid_unique = train_test_split(
            sequences,
            test_size=config.get("valid_subset_of_train"),  # 划分比例
            random_state=42,
        )
        train_unique = set(train_unique)
        valid_unique = set(valid_unique)

        # 根据划分好的序列集合，过滤训练和验证数据框
        train_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in train_unique)
        valid_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in valid_unique)

        # Save splits
        # TODO: Optionally load the data splits
        # TODO: Allow loading of data splits in `predict.py`
        # TODO: Upload to Aichor
        split_path = os.path.join(
            config.get("model_save_folder_path", "./checkpoints"), "splits.csv"
        )
        os.makedirs(os.path.dirname(split_path), exist_ok=True)

        # 将划分结果保存到 CSV 文件，以备后续使用
        pd.DataFrame(
            {
                "modified_sequence": list(train_unique) + list(valid_unique),
                "split": ["train"] * len(train_unique) + ["valid"] * len(valid_unique),
            }
        ).to_csv(str(split_path), index=False)
        logger.info(f"Data splits saved to {split_path}")

    # Check residues
    # 执行数据完整性检查
    if config.get("perform_data_checks", True):
        # 检查数据中是否存在模型词汇表之外的氨基酸/修饰，并过滤掉这些行
        logger.info(f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows.")
        # ... (词汇表检查与过滤逻辑)
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        data_residues = set()
        data_residues.update(train_sdf.get_vocabulary(residue_set.tokenize))
        data_residues.update(valid_sdf.get_vocabulary(residue_set.tokenize))
        if len(data_residues - supported_residues) > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            logger.info(f"New residues found: \n{data_residues - supported_residues}")
            logger.info(f"Residues supported: \n{supported_residues}")
            original_size = (len(train_sdf), len(valid_sdf))
            train_sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            valid_sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            new_size = (len(train_sdf), len(valid_sdf))
            logger.warning(
                f"{original_size[0] - new_size[0]:,d} "
                f"({(original_size[0] - new_size[0]) / original_size[0] * 100:.2f}%) "
                "training rows dropped."
            )
            logger.warning(
                f"{original_size[1] - new_size[1]:,d} "
                f"({(original_size[1] - new_size[1]) / original_size[1] * 100:.2f}%) "
                "validation rows dropped."
            )

        # Check charge values:
        # 检查并过滤掉电荷数超出模型支持范围的数据行
        original_size = (len(train_sdf), len(valid_sdf))
        train_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
                        and (row["precursor_charge"] > 0)
        )
        if len(train_sdf) < original_size[0]:
            logger.warning(
                f"Found {original_size[0] - len(train_sdf)} rows in training set with charge > "
                f"{config.get('max_charge', 10)} or <= 0. These rows will be skipped."
            )

        valid_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
                        and (row["precursor_charge"] > 0)
        )
        if len(valid_sdf) < original_size[1]:
            logger.warning(
                f"Found {original_size[1] - len(valid_sdf)} rows in training set with charge > "
                f"{config.get('max_charge', 10)}. These rows will be skipped."
            )

    # 根据配置对训练集和验证集进行采样，方便快速实验
    train_sdf.sample_subset(fraction=config.get("train_subset", 1.0), seed=42)
    valid_sdf.sample_subset(fraction=config.get("valid_subset", 1.0), seed=42)

    # --- 5. PyTorch 数据集与加载器创建 ---
    # 将处理好的数据封装成 PyTorch Dataset 对象，用于将数据转换为模型可接受的张量
    train_ds = SpectrumDataset(
        train_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        peptide_pad_length=config.get("max_length", 40)
        if config.get("compile_model", False)
        else 0,
        pad_spectrum_max_length=config.get("compile_model", False)
                                or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )

    valid_ds = SpectrumDataset(
        valid_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        pad_spectrum_max_length=config.get("compile_model", False)
                                or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )

    logger.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )

    train_sequences = pl.Series(list(train_sdf.get_unique_sequences()))
    valid_sequences = pl.Series(list(valid_sdf.get_unique_sequences()))

    # Check warmup
    # 检查学习率预热（warmup）的步数是否合理
    if config.get("warmup_iters", 100_000) > len(train_ds) / config.get("train_batch_size", 256):
        logger.warning(
            "Model warmup is greater than one epoch of the training set. "
            "Check warmup_iters in config"
        )

    # Check how many times model will save
    # 检查模型保存相关的配置是否合理
    if config.get("save_model", True):
        total_epochs = config.get("epochs", 30)
        epochs_per_save = 1 / (
                len(train_ds) / config.get("train_batch_size", 256) / config.get("ckpt_interval")
        )
        if epochs_per_save > total_epochs:
            logger.warning(
                f"Model checkpoint will never save. Attempting to save every {epochs_per_save:.2f} "
                f"epochs but only training for {total_epochs:d} epochs. "
                "Check ckpt_interval in config."
            )
        else:
            logger.info(f"Model checkpointing every {epochs_per_save:.2f} epochs.")

    # 检查训练集是否与黑名单序列重叠
    if config.get("blacklist", None):
        logger.info("Checking if any training set overlaps with blacklisted sequences...")
        blacklist_df = pd.read_csv(config["blacklist"])
        leakage = any(
            train_sequences.map_elements(remove_modifications, return_dtype=pl.String).is_in(
                blacklist_df["sequence"]
            )
        )
        if leakage:
            raise ValueError(
                "Portion of training set sequences overlaps with blacklisted sequences."
            )
        else:
            logger.info("No blacklisted sequences!")

    # 检查验证集是否与训练集重叠（数据泄露）
    if config.get("perform_data_checks", True):
        logger.info("Checking if any validation set overlaps with training set...")
        leakage = any(valid_sequences.is_in(train_sequences))
        if leakage:
            raise ValueError("Portion of validation set sequences overlaps with training set.")
        else:
            logger.info("No data leakage!")

    # Modified code
    train_ds_for_processing = SpectrumDataset(
        train_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        peptide_pad_length=config.get("max_length", 40),
        pad_spectrum_max_length=config.get("compile_model", False) or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )

    # test
    valid_ds_for_processing = SpectrumDataset(
        valid_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        pad_spectrum_max_length=config.get("compile_model", False) or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
    )
    # logger.info("Sequencing the SpectrumDataset to list")
    # train_cache = [train_ds_for_processing[i] for i in range(len(train_ds_for_processing))]
    # valid_cache = [valid_ds_for_processing[i] for i in range(len(valid_ds_for_processing))]
    # if config['save_readyData']:
    #     torch.save(train_cache, config['train_readyData'])
    #     torch.save(valid_cache, config['test_readyData'])
    #
    # # 4. 释放不再需要的、占用大量内存的原始对象的内存
    # del train_sdf, valid_sdf, train_ds_for_processing, valid_ds_for_processing
    # logger.info("Pre-processing and caching complete. Original dataframes have been released from memory.")

    # 5. 直接将缓存好的列表传递给 DataLoader
    #    PyTorch 的 DataLoader 非常智能，可以直接处理一个列表作为数据集
    import multiprocessing as mp
    # train_dl = DataLoader(
    #     train_ds_for_processing,  # <-- 直接传入列表！
    #     batch_size=config["train_batch_size"],
    #     num_workers=0,  # <-- 现在可以安全地使用多个 worker 了！
    #     shuffle=True,  # <-- DataLoader 可以安全地对列表进行 shuffle
    #     collate_fn=collate_batch,
    #     pin_memory=True  # 建议开启，可以加速数据到 GPU 的传输
    # )
    # valid_dl = DataLoader(
    #     valid_ds_for_processing,  # <-- 直接传入列表！
    #     batch_size=config["predict_batch_size"],
    #     num_workers=0,
    #     shuffle=False,
    #     collate_fn=collate_batch,
    #     pin_memory=True
    # )
    train_dl = DataLoader(
        train_ds_for_processing,
        batch_size=config["train_batch_size"],
        num_workers=config["num_workers"],
        multiprocessing_context=mp.get_context("spawn"),
        shuffle=True,  # 先关掉，确认能跑通
        pin_memory=True,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=True,
        collate_fn=collate_batch,
    )

    valid_dl = DataLoader(
        valid_ds_for_processing,  # <-- 直接传入列表！
        batch_size=config["predict_batch_size"],
        multiprocessing_context=mp.get_context("spawn"),
        num_workers=config["num_workers"],
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=config["prefetch_factor"],
        collate_fn=collate_batch,
        pin_memory=True
    )

    # 将 Dataset 封装成 DataLoader，用于高效地进行数据批处理
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=config["train_batch_size"],
    #     num_workers=0 if config['lazy_loading'] else 6,  # SDF requirement is 0
    #     # num_workers=0,
    #     shuffle=False,  # SDF requirement
    #     collate_fn=collate_batch,
    # )
    #
    # valid_dl = DataLoader(
    #     valid_ds,
    #     batch_size=config["predict_batch_size"],
    #     num_workers=0 if config['lazy_loading'] else 6,  # SDF requirement is 0
    #     # num_workers=0,
    #     shuffle=False,
    #     collate_fn=collate_batch,
    # )

    # Update rates based on bs=32
    step_scale = 32 / config["train_batch_size"]
    logger.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    # 打印一个样本批次的信息，用于调试
    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch
    logger.info("Sample batch:")
    logger.info(f" - spectra.shape={spectra.shape}")
    logger.info(f" - precursors.shape={precursors.shape}")
    logger.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logger.info(f" - peptides.shape={peptides.shape}")
    logger.info(f" - peptides_mask.shape={peptides_mask.shape}")

    # init model
    # --- 6. 模型初始化与检查点加载 ---

    # 根据配置初始化 InstaNovo 模型架构
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

    # 判断是否从头训练，如果不是，则设置检查点路径
    if not config["train_from_scratch"]:
        resume_checkpoint_path = config["resume_checkpoint"]
    else:
        resume_checkpoint_path = None

    # 如果指定了检查点路径，则加载预训练权重
    if resume_checkpoint_path is not None:
        logger.info(f"Loading model checkpoint from '{resume_checkpoint_path}'")
        model_state = torch.load(resume_checkpoint_path, map_location="cpu")
        # check if PTL checkpoint
        # 兼容 PyTorch Lightning 保存的检查点（移除 "model." 前缀）
        if "state_dict" in model_state:
            model_state = {k.replace("model.", ""): v for k, v in model_state["state_dict"].items()}

        # 核心兼容性处理：检查检查点的词汇表大小是否与当前配置匹配
        aa_embed_size = model_state["head.weight"].shape[0]
        if aa_embed_size != len(residue_set):
            state_keys = ["head.weight", "head.bias", "aa_embed.weight"]
            logger.warning(
                f"Model expects vocab size of {len(residue_set)}, checkpoint has {aa_embed_size}."
            )
            # 如果不匹配，根据配置的冲突解决策略（删除/随机初始化/部分保留）来调整权重矩阵
            # 这使得可以在不同词汇表上进行微调
            logger.warning("Assuming a change was made to the residues in the configuration file.")
            logger.warning(f"Automatically converting {state_keys} to match expected.")

            new_model_state = model.state_dict()

            resolution = config.get("residue_conflict_resolution", "delete")
            for k in state_keys:
                # initialise weights to normal distribution with weight 1/sqrt(dim)
                tmp = torch.normal(
                    mean=0,
                    std=1.0 / np.sqrt(config["dim_model"]),
                    size=new_model_state[k].shape,
                    dtype=new_model_state[k].dtype,
                )
                if "bias" in k:
                    # initialise bias to zeros
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

            logger.warning(
                f"Model checkpoint has {len(state_keys)} weights updated with '{resolution}' "
                "conflict resolution"
            )

        k_missing: int = np.sum(
            [x not in list(model_state.keys()) for x in list(model.state_dict().keys())]
        )
        if k_missing > 0:
            logger.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing: int = np.sum(
            [x not in list(model.state_dict().keys()) for x in list(model_state.keys())]
        )
        if k_missing > 0:
            logger.warning(f"Model state is missing {k_missing} keys!")

        # 加载权重，strict=False 允许部分键不匹配
        model.load_state_dict(model_state, strict=False)

    logger.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # --- 7. 训练组件初始化 ---
    if not config["conv_peak_encoder"]:
        logger.info("Test forward pass:")
        with torch.no_grad():
            y = model(spectra, precursors, peptides, spectra_mask, peptides_mask)
            logger.info(f" - y.shape={y.shape}")

    # Set device to train on
    # 将模型移动到指定设备
    device = check_device(config=config)
    model = model.to(device)

    if config.get("fp16", True) and device.lower() == "cpu":
        logger.warning("fp16 is enabled but device type is cpu. fp16 will be disabled.")
        config["fp16"] = False

    # 初始化用于验证的解码器和指标计算器
    decoder = GreedyDecoder(model=model)  # 此处用的是运行速度最快的贪婪解码器
    metrics = Metrics(residue_set, config["isotope_error_range"])

    # 初始化优化器 (Adam) 和自定义的学习率调度器 (WarmupScheduler)
    # Use as an additional data sanity check
    if config.get("validate_precursor_mass", True):
        logger.info("Sanity checking precursor masses for training set...")
        train_sdf.validate_precursor_mass(metrics)
        logger.info("Sanity checking precursor masses for validation set...")
        valid_sdf.validate_precursor_mass(metrics)

    # init optim
    # 初始化优化器 (Adam) 和自定义的学习率调度器 (WarmupScheduler)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = WarmupScheduler(optim, config["warmup_iters"])
    # 获取分布式训练策略
    strategy = _get_strategy()

    # 如果有分组验证，准备分组信息
    if validation_group_mapping is not None:
        logger.info("Computing validation groups.")
        validation_groups = pl.Series(
            [
                validation_group_mapping[row["source_file"]]
                if row.get("source_file", None) in validation_group_mapping
                else "no_group"
                for row in iter(valid_sdf)
            ],
            dtype=pl.String,
        )
        logger.info("Sequences per validation group:")
        for group in validation_groups.unique():
            logger.info(f" - {group}: {(validation_groups == group).sum():,d}")
    else:
        validation_groups = None

    # --- 8. PyTorch Lightning Trainer 设置与启动 ---

    # 将所有训练组件（模型、数据、优化器等）封装到之前定义的 PTModule 中
    ptmodel = PTModule(
        config,
        model,
        decoder,
        metrics,
        sw,
        optim,
        scheduler,
        config["compile_model"],
        config["fp16"],
        validation_groups,
    )

    # 2. 实例化 Profiler，并进行详细配置
    #    这里的 profiler_log_dir 可以是你希望保存分析结果的任何地方
    if config["profiler"]:
        profiler_log_dir = os.path.join(config["tb_summarywriter"], "profiler")
        logger.info(f"Profiler trace will be saved to: {profiler_log_dir}")

        profiler = PyTorchProfiler(
            dirpath=profiler_log_dir,
            filename="profile_trace",  # 生成的文件会是 profile_trace.pt.trace.json
            # 使用 schedule 来精确控制分析的 step
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=2),
            # 自动处理和保存 trace 文件
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
            # 开启其他有用的选项
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    # 配置模型保存的回调（Callback）
    if config["save_model"]:
        logger.info("Model saving enabled")

        # 根据是否在 Aichor 平台，选择使用自定义的 S3 保存 Wrapper 或标准的 ModelCheckpoint
        # returns input if s3 disabled
        s3_ckpt_path = S3FileHandler.convert_to_s3_output(config["model_save_folder_path"])
        if S3FileHandler._aichor_enabled():
            callbacks = [
                PLCheckpointWrapper(
                    dirpath=config["model_save_folder_path"],
                    save_top_k=-1,
                    save_weights_only=config["save_weights_only"],
                    every_n_train_steps=config["ckpt_interval"],
                    s3_ckpt_path=s3_ckpt_path,
                    s3=s3,
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

    # 初始化 PyTorch Lightning 的 Trainer，它将自动处理所有训练循环的细节
    logger.info("Initializing Pytorch Lightning trainer.")
    trainer = L.pytorch.Trainer(
        profiler=profiler if config["profiler"] else None,
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
        strategy=strategy
    )

    # Train the model.
    # 启动训练！这一行代码会启动 PyTorch Lightning 的训练和验证循环
    logger.info("InstaNovo training started.")
    trainer.fit(ptmodel, train_dl, valid_dl)

    logger.info("InstaNovo training finished.")


def _get_strategy() -> DDPStrategy | str:
    """Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns:
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    """
    为 PyTorch Lightning Trainer 获取分布式训练策略。

    DDP (分布式数据并行) 策略在有多于一个 GPU 时效果最好。
    它在只有 CPU 时也能工作，但在苹果芯片 (MPS) 上会因 Gloo 后端不兼容而失败。
    """
    # 检查可用的 CUDA (NVIDIA GPU) 设备数量是否大于 1
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return "auto"


def _set_author_neptune_api_token() -> None:
    """Set the variable NEPTUNE_API_TOKEN based on the email of commit author.

    It is useful on AIchor to have proper owner of each run.
    """
    """
    一个非常特殊的函数，用于在 Aichor 云计算平台上运行时，
    根据 Git 提交作者的邮箱，自动设置 Neptune 实验跟踪工具的 API 密钥。
    这确保了每个实验运行都能正确地归属于其作者。
    """
    try:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
    # we are not on AIchor
    except KeyError:
        logger.debug(
            "We are not running on AIchor (https://aichor.ai/), not looking for Neptune API token."
        )
        return

    author_email, _ = author_email.split("@")
    author_email = author_email.replace("-", "_").replace(".", "_").upper()

    logger.info(f"Checking for Neptune API token under {author_email}__NEPTUNE_API_TOKEN.")
    try:
        author_api_token = os.environ[f"{author_email}__NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_API_TOKEN"] = author_api_token
        logger.info(f"Set token for {author_email}.")
    except KeyError:
        logger.info(f"Neptune credentials for user {author_email} not found.")


class NeptuneSummaryWriter(SummaryWriter):
    """Combine SummaryWriter with NeptuneWriter."""
    """
    一个自定义的 SummaryWriter 类，它继承自 PyTorch 的标准 SummaryWriter。
    作用是将其功能与 Neptune 的日志记录功能结合起来，实现一次调用，双重记录。
    """

    def __init__(self, log_dir: str, run: neptune.Run) -> None:
        super().__init__(log_dir=log_dir)
        self.run = run

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        """Record scalar to tensorboard and Neptune."""
        super().add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self.run[tag].append(scalar_value, step=global_step)


def _format_time(seconds: float) -> str:
    """一个简单的时间格式化工具，将秒数转换为 HH:MM:SS 格式的字符串。"""
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _get_filepath_mapping(file_groups: dict[str, str]) -> dict[str, str]:
    """
    为分组验证数据创建一个从“单个文件路径”到“组名”的映射字典。
    """
    group_mapping = {}
    for group, path in file_groups.items():
        for fp in SpectrumDataFrame._convert_file_paths(path):
            group_mapping[fp] = group
    return group_mapping


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""
    """
    一个自定义的学习率调度器，实现了线性预热 (Linear Warmup) 功能。
    它继承自 PyTorch 的基础学习率调度器类。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """PyTorch 优化器会调用此方法来获取当前步骤的学习率。"""
        """Get the learning rate at the current step."""
        # 首先获取一个学习率缩放因子
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        # 将基础学习率乘以这个因子，得到当前的学习率
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """计算当前步骤的学习率缩放因子。"""
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        # 如果当前步数小于等于预热总步数
        if epoch <= self.warmup:
            # 学习率因子从 0 线性增加到 1.0
            lr_factor *= epoch / self.warmup
        return lr_factor


# --- 脚本主入口 ---
# @hydra.main 是一个装饰器，它将 main 函数指定为脚本的入口点，并使用 Hydra 来管理配置
# TODO remove main function
@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="instanovo")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # 调用前面定义的特殊函数，尝试设置 Neptune API Token
    _set_author_neptune_api_token()

    # Unnest hydra configs

    # “解包” Hydra 的嵌套配置。
    # 这是一个便利操作，将 config.model.dim_model 这样的嵌套结构，扁平化为 config.dim_model
    # sub_configs_list = ["model", "dataset", "residues"]
    # for sub_name in sub_configs_list:
    #     if sub_name in config:
    #         with open_dict(config): # 允许修改配置对象
    #             temp = config[sub_name]
    #             del config[sub_name]
    #             config.update(temp)

    sub_configs_list = ["model", "dataset"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):  # 允许修改配置对象
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"Imported hydra config:\n{OmegaConf.to_yaml(config)}")


    # 调用之前定义的总控函数，传入最终处理好的配置，正式开始训练流程
    train(config)


if __name__ == "__main__":
    main()
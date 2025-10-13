# utils/spectrum_iterable_dataset.py
from __future__ import annotations
import math
import os
import random
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
import spectrum_utils.spectrum as sus

# 复用你仓库里的引用
from ..__init__ import console
from ..constants import ANNOTATED_COLUMN
from ..types import Peptide, Spectrum
from ..utils import ResidueSet
from ..utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


# --------------------------- DDP / 并发辅助 ---------------------------

def _get_rank_world_size() -> tuple[int, int]:
    """优先用 torch.distributed，回退到环境变量；单进程则 (0,1)。"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass

    def _int_env(keys: List[str], default: int) -> int:
        for k in keys:
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v)
                except ValueError:
                    continue
        return default

    rank = _int_env(["RANK", "LOCAL_RANK"], 0)
    world = _int_env(["WORLD_SIZE", "LOCAL_WORLD_SIZE"], 1)
    return rank, world


def _resolve_arrow_files(data_path: str | Path, split: str) -> List[Tuple[Path, str]]:
    """列出所有属于 split 的 Arrow/Parquet 分片文件。"""
    data_path = Path(data_path)
    if data_path.is_file():
        suf = data_path.suffix.lower()
        fmt = "ipc" if suf == ".ipc" else "parquet"
        return [(data_path, fmt)]

    files: List[Tuple[Path, str]] = []
    for p in sorted(data_path.iterdir()):
        suf = p.suffix.lower()
        stem = p.stem.lower()
        if suf not in (".ipc", ".parquet"):
            continue
        # 支持以下命名：
        # 1) {split}_shard*.{ipc|parquet}
        # 2) dataset-*-{split}-*.{ipc|parquet}  ← SpectrumDataFrame.save 的格式
        # 3) {split}.{ipc|parquet}
        if (f"{split}_shard" in stem) or (f"-{split}-" in stem) or (stem == split):
            files.append((p, "ipc" if suf == ".ipc" else "parquet"))

    if not files:
        fb_ipc = data_path / f"{split}.ipc"
        fb_parq = data_path / f"{split}.parquet"
        if fb_ipc.exists():
            files = [(fb_ipc, "ipc")]
        elif fb_parq.exists():
            files = [(fb_parq, "parquet")]
        else:
            raise ValueError(
                f"在 {data_path} 未找到分片或回退文件（{split}.ipc / {split}.parquet / dataset-*-{split}-*.parquet）。"
            )

    return files


# --------------------------- 清洗 / 标准化 ---------------------------

def _clean_and_remap_batch(tbl: pa.Table) -> pl.DataFrame:
    df = pl.from_arrow(tbl)

    # 把 ANNOTATED_COLUMN 标准化到 modified_sequence
    rename_map: dict[str, str] = {}
    if ANNOTATED_COLUMN in df.columns and "modified_sequence" not in df.columns:
        rename_map[ANNOTATED_COLUMN] = "modified_sequence"

    # 常见列名映射
    col_map: dict[str, str] = {
        "Modified sequence": "modified_sequence",
        "MS/MS m/z": "precursor_mz",
        "Precursor m/z": "precursor_mz",
        "Theoretical m/z": "theoretical_mz",
        "Mass": "precursor_mass",
        "Charge": "precursor_charge",
        "Mass values": "mz_array",
        "Mass spectrum": "mz_array",
        "Intensity": "intensity_array",
        "Raw intensity spectrum": "intensity_array",
    }
    rename_map.update({k: v for k, v in col_map.items() if k in df.columns})
    if rename_map:
        df = df.rename(rename_map)

    # 只保留必要列（若需要 group/source_file 也保留）
    keep_cols = ["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"]
    for extra in ("source_file", "group"):
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    df = df.drop([c for c in df.columns if c not in keep_cols and c not in ("theoretical_mz", "precursor_mass")])

    # 统一 dtype
    wanted: dict[str, pl.DataType] = {
        "modified_sequence": pl.Utf8,
        "precursor_mz": pl.Float64,
        "precursor_charge": pl.Int32,
        "mz_array": pl.List(pl.Float32),
        "intensity_array": pl.List(pl.Float32),
    }
    for k, t in wanted.items():
        if k in df.columns:
            df = df.with_columns(pl.col(k).cast(t))

    # 必要列校验
    for k in ["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"]:
        if k not in df.columns:
            raise KeyError(f"缺失必要列: {k}")

    # 有些数据集序列两端可能有占位符（如 '.' 或 '_'）
    if df.height > 0:
        first_val = df.select(pl.first("modified_sequence")).item()
        if isinstance(first_val, str) and len(first_val) > 0 and first_val[0] in (".", "_"):
            df = df.with_columns(
                pl.col("modified_sequence").map_elements(lambda x: x[1:-1], return_dtype=pl.Utf8)
            )

    # 输出列顺序
    out_cols = ["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"]
    for extra in ("group", "source_file"):
        if extra in df.columns:
            out_cols.append(extra)
    return df.select(out_cols)


# --------------------------- Dataset ---------------------------

class SpectrumIterableDataset(IterableDataset):
    """
    Arrow/Parquet 流式分片读取的 IterableDataset：
    - 正确的 rank/worker 切分顺序（先 rank 后 worker），保证 DDP 下各 rank 均有数据；
    - 当分片数量 < (world_size * num_workers) 时，采用“复制分配”以避免某些 rank/worker 空迭代；
    - 可选大窗口缓冲打乱（shuffle_buffer）；
    - 支持 spectrum_utils 预处理路径或快速张量路径；
    - 可在需要时把谱图 pad/截断为固定长度（配合 torch.compile）。
    """

    def __init__(
        self,
        data_path: str | Path,
        residue_set: ResidueSet,
        split: str = "train",
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        pad_spectrum_max_length: bool = False,
        peptide_pad_length: int = 30,
        reverse_peptide: bool = True,
        annotated: bool = True,
        return_str: bool = False,
        bin_spectra: bool = False,
        bin_size: float = 0.01,
        add_eos: bool = True,
        tokenize_peptide: bool = True,
        # iterable 特有
        batch_rows: int = 65536,
        shuffle_buffer: int = 0,
        use_sus_preprocess: bool = True,
        # 读 Arrow scanner 是否多线程
        use_threads: bool = True,
        # 是否跨 rank 分片（验证可关闭）
        shard_across_ranks: bool = True,
        # ---- 新增：用于验证分组 ----
        return_group: bool = False,
        group_col: str = "source_file",
        group_mapping: Optional[Dict[str, str]] = None,
        shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = str(data_path)
        self.residue_set = residue_set
        self.split = split

        self.n_peaks = int(n_peaks)
        self.min_mz = float(min_mz)
        self.max_mz = float(max_mz)
        self.remove_precursor_tol = float(remove_precursor_tol)
        self.min_intensity = float(min_intensity)
        self.pad_spectrum_max_length = bool(pad_spectrum_max_length)
        self.peptide_pad_length = int(peptide_pad_length)
        self.reverse_peptide = bool(reverse_peptide)
        self.annotated = bool(annotated)
        self.return_str = bool(return_str)
        self.bin_spectra = bool(bin_spectra)
        self.bin_size = float(bin_size)
        self.add_eos = bool(add_eos)
        self.tokenize_peptide = bool(tokenize_peptide)

        self.batch_rows = int(batch_rows)
        self.shuffle_buffer = int(shuffle_buffer)
        self.use_sus_preprocess = bool(use_sus_preprocess)
        self.use_threads = bool(use_threads)
        self.shard_across_ranks = bool(shard_across_ranks)

        # 新增
        self.return_group = bool(return_group)
        self.group_col = str(group_col)
        self.group_mapping = group_mapping
        self.shuffle_files = bool(shuffle_files)

        self._files_with_fmt = _resolve_arrow_files(self.data_path, self.split)  # List[(Path, str)]

        if self.bin_spectra:
            # 预生成直方图 bin 边界（float32）
            nbins = int(np.floor(self.max_mz / self.bin_size)) + 1
            self.bins = torch.linspace(
                0, self.bin_size * nbins, steps=nbins + 1, dtype=torch.float32
            )
        else:
            self.bins = None

    # --------------------------- 迭代主逻辑 ---------------------------

    def __iter__(self) -> Iterator[Tuple[Spectrum, float, int, Peptide | str] | Tuple[Spectrum, float, int, Peptide | str, str]]:
        # 并发单位：rank * num_workers（若不跨 rank，则 world_size=1）
        wi = get_worker_info()
        rank, world_size = _get_rank_world_size()
        if not self.shard_across_ranks:
            rank, world_size = 0, 1

        num_workers = wi.num_workers if wi is not None else 0
        worker_id = wi.id if wi is not None else 0

        n_units = world_size * (num_workers if num_workers > 0 else 1)
        unit_id = rank * (num_workers if num_workers > 0 else 1) + worker_id

        # 局部 RNG：保证并发环境下可复现的 shuffle / 采样
        if wi is not None and hasattr(wi, "seed"):
            seed = int(wi.seed)
        else:
            seed = int(torch.initial_seed()) ^ (rank << 16) ^ (worker_id + 0x9E3779B1)
        rng = random.Random(seed)

        files = list(self._files_with_fmt)
        n_files = len(files)

        # 分配策略
        if n_files >= n_units:
            assigned = files[unit_id::n_units]
        else:
            reps = math.ceil(n_units / max(n_files, 1))
            repeated = (files * reps)[:n_units]
            assigned = [repeated[unit_id]]

        # 验证集通常不希望打乱分片顺序；训练集可保持随机
        if self.shuffle_files:
            rng.shuffle(assigned)

        if not assigned:
            return iter(())

        # 大窗口打乱缓冲
        buffer: List[Tuple[Spectrum, float, int, Peptide | str] | Tuple[Spectrum, float, int, Peptide | str, str]] = []

        for fpath, fmt in assigned:
            dataset = ds.dataset(str(fpath), format=fmt)
            scanner = dataset.scanner(filter=None, use_threads=self.use_threads, batch_size=self.batch_rows)

            for rb in scanner.to_batches():
                tbl = pa.Table.from_batches([rb])
                try:
                    df_batch = _clean_and_remap_batch(tbl)
                except Exception as e:
                    logger.warning(f"批次清洗失败（{fpath.name}）：{e}; 跳过该批。")
                    continue
                if df_batch.height == 0:
                    continue

                # 遍历行构建样本
                for row in df_batch.iter_rows(named=True):
                    try:
                        item = self._build_item_from_row(row)
                    except Exception as e:
                        logger.debug(f"行处理失败：{e}; 使用占位样本。")
                        item = self._dummy_item()

                    if self.shuffle_buffer > 0:
                        buffer.append(item)
                        if len(buffer) >= self.shuffle_buffer:
                            j = rng.randrange(len(buffer))
                            yield buffer.pop(j)
                    else:
                        yield item

        # 清空缓冲区（保持打乱）
        while buffer:
            j = rng.randrange(len(buffer))
            yield buffer.pop(j)

    # --------------------------- 样本构造 / 预处理 ---------------------------

    def _dummy_item(self):
        spectrum = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        precursor_mz = 1.0
        precursor_charge = 1
        if self.return_str:
            pep_out: Peptide | str = ""
        else:
            pep_out = torch.zeros(
                (self.peptide_pad_length + (1 if self.add_eos else 0),), dtype=torch.long
            )
        if not self.return_group:
            return spectrum, precursor_mz, precursor_charge, pep_out
        return spectrum, precursor_mz, precursor_charge, pep_out, "no_group"

    def _build_item_from_row(self, row: dict):
        # 转 dtype（尽量零拷贝）
        mz_np = np.asarray(row["mz_array"], dtype=np.float32)
        int_np = np.asarray(row["intensity_array"], dtype=np.float32)
        mz = torch.from_numpy(mz_np)
        inten = torch.from_numpy(int_np)

        precursor_mz: float = float(row["precursor_mz"])
        precursor_charge: int = int(row["precursor_charge"])

        peptide_str: str = row["modified_sequence"] if self.annotated else ""

        spectrum = self._process_peaks(mz, inten, precursor_mz, precursor_charge)

        if self.bin_spectra:
            # 1D 直方图
            hist = torch.histogram(spectrum[:, 0], bins=self.bins, weight=spectrum[:, 1]).hist  # type: ignore
            spectrum = hist  # shape: (nbins,)
        elif self.pad_spectrum_max_length:
            # 固定长度 (n_peaks, 2)
            if spectrum.shape[0] > self.n_peaks:
                spectrum = spectrum[: self.n_peaks]
            if spectrum.shape[0] < self.n_peaks:
                out = torch.zeros((self.n_peaks, 2), dtype=spectrum.dtype)
                out[: spectrum.shape[0]] = spectrum
                spectrum = out

        if self.return_str:
            pep_out: Peptide | str = peptide_str
        else:
            if self.tokenize_peptide:
                pep_tokens = self.residue_set.tokenize(peptide_str)
            else:
                pep_tokens = peptide_str  # type: ignore
            if self.reverse_peptide:
                pep_tokens = pep_tokens[::-1]
            pep_ids = self.residue_set.encode(pep_tokens, add_eos=self.add_eos, return_tensor="pt")

            pad_len = self.peptide_pad_length + (1 if self.add_eos else 0)
            L = min(pad_len, int(pep_ids.shape[0]))
            pep_pad = torch.zeros((pad_len,), dtype=pep_ids.dtype)
            pep_pad[:L] = pep_ids[:L]
            pep_out = pep_pad

        if not self.return_group:
            return spectrum, precursor_mz, precursor_charge, pep_out

        # ---- 附带 group 信息 ----
        g = str(row.get("group", row.get(self.group_col, "")))
        if self.group_mapping is not None and g:
            g = self.group_mapping.get(g, "no_group")
        if not g:
            g = "no_group"
        return spectrum, precursor_mz, precursor_charge, pep_out, g

    def _process_peaks(
        self,
        mz_array: Tensor,          # (P,)
        int_array: Tensor,         # (P,)
        precursor_mz: float,
        precursor_charge: int,
    ) -> Spectrum:
        """与原版等价的谱图预处理；默认使用 spectrum_utils，可切换纯张量快路径。"""
        if self.use_sus_preprocess:
            mz_np = mz_array.detach().cpu().numpy().astype(np.float32, copy=False)
            int_np = int_array.detach().cpu().numpy().astype(np.float32, copy=False)
            spec = sus.MsmsSpectrum(
                identifier="",
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                mz=mz_np,
                intensity=int_np,
            )
            try:
                spec.set_mz_range(self.min_mz, self.max_mz)
                if len(spec.mz) == 0:
                    raise ValueError
                spec.remove_precursor_peak(self.remove_precursor_tol, "Da")
                if len(spec.mz) == 0:
                    raise ValueError
                spec.filter_intensity(self.min_intensity, self.n_peaks)
                if len(spec.mz) == 0:
                    raise ValueError
                spec.scale_intensity("root", 1)
                norm = np.linalg.norm(spec.intensity)
                if norm == 0:
                    return torch.tensor([[0.0, 1.0]], dtype=torch.float32)
                intensities = spec.intensity / norm
                arr = torch.tensor(np.array([spec.mz, intensities]), dtype=torch.float32).T
                return arr
            except ValueError:
                return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        # 纯张量快路径
        if int_array.numel() == 0:
            return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        maxv = torch.max(int_array)
        if maxv.item() > 0:
            int_array = int_array / maxv

        bad = (int_array < self.min_intensity) | (mz_array < self.min_mz) | (mz_array > self.max_mz)
        keep = ~bad
        mz_array = mz_array[keep]
        int_array = int_array[keep]

        if mz_array.numel() == 0:
            return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        if self.n_peaks < int_array.numel():
            topk = torch.topk(int_array, k=self.n_peaks, largest=True, sorted=False)
            idx = topk.indices
            mz_array = mz_array[idx]
            int_array = int_array[idx]

        return torch.stack([mz_array, int_array], dim=1).to(torch.float32)

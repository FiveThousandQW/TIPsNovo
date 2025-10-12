# utils/spectrum_iterable_dataset.py
from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch import Tensor
import spectrum_utils.spectrum as sus
from torch.utils.data import IterableDataset

# 复用你仓库里的引用
from ..__init__ import console
from ..constants import ANNOTATED_COLUMN
from ..types import Peptide, Spectrum
from ..utils import ResidueSet
from ..utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

def _resolve_arrow_files(data_path: str | Path, split: str) -> List[Tuple[Path, str]]:
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
        # 兼容三类命名：
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
            raise ValueError(f"在 {data_path} 未找到分片或回退文件（{split}.ipc / {split}.parquet / dataset-*-{split}-*.parquet）。")

    return files



def _clean_and_remap_batch(tbl: pa.Table) -> pl.DataFrame:
    df = pl.from_arrow(tbl)

    # 先把 ANNOTATED_COLUMN 映射回 modified_sequence（若必要）
    rename_map = {}
    if ANNOTATED_COLUMN in df.columns and "modified_sequence" not in df.columns:
        rename_map[ANNOTATED_COLUMN] = "modified_sequence"

    # 兼容各种导出列名
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

    # 只保留训练需要列
    keep_cols = ["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"]
    df = df.drop([c for c in df.columns if c not in keep_cols and c not in ("theoretical_mz", "precursor_mass")])

    # 规范 dtype
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

    # 必要列检查
    for k in ["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"]:
        if k not in df.columns:
            raise KeyError(f"缺失必要列: {k}")

    # 去掉某些数据集里序列两端的占位符（看首字符）
    if df.height > 0:
        first_val = df.select(pl.first("modified_sequence")).item()
        if isinstance(first_val, str) and len(first_val) > 0 and first_val[0] in (".", "_"):
            df = df.with_columns(
                pl.col("modified_sequence").map_elements(lambda x: x[1:-1], return_dtype=pl.Utf8)
            )

    return df.select(["modified_sequence", "precursor_mz", "precursor_charge", "mz_array", "intensity_array"])


class SpectrumIterableDataset(IterableDataset):
    """
    直接替换版：支持基于 Arrow 的流式/分片读取与并行预处理。
    - 每个 worker 处理不同的 shard 列表；
    - 通过 batch_rows 控制 Arrow 扫描批大小；
    - shuffle_buffer>0 时使用 reservoir/缓冲打乱。
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
        # 读 Arrow scanner 时是否并行
        use_threads: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = str(data_path)
        self.residue_set = residue_set
        self.split = split

        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.pad_spectrum_max_length = pad_spectrum_max_length
        self.peptide_pad_length = peptide_pad_length
        self.reverse_peptide = reverse_peptide
        self.annotated = annotated
        self.return_str = return_str
        self.bin_spectra = bin_spectra
        self.bin_size = bin_size
        self.add_eos = add_eos
        self.tokenize_peptide = tokenize_peptide

        self.batch_rows = int(batch_rows)
        self.shuffle_buffer = int(shuffle_buffer)
        self.use_sus_preprocess = use_sus_preprocess
        self.use_threads = use_threads

        self._files_with_fmt = _resolve_arrow_files(self.data_path, self.split)  # List[(Path, str)]

        if self.bin_spectra:
            # 预生成直方图 bin 边界（float32）
            nbins = int(np.floor(self.max_mz / self.bin_size)) + 1
            self.bins = torch.linspace(0, self.bin_size * nbins, steps=nbins + 1, dtype=torch.float32)
        else:
            self.bins = None

    # --------------------------- 核心迭代 ---------------------------

    def __iter__(self) -> Iterator[Tuple[Spectrum, float, int, Peptide | str]]:

        """
        基于 Arrow 扫描的流式读取：
        - 按 worker 拆分文件子集（避免文件争用）
        - 每个文件用正确的 format ('ipc' 或 'parquet') 构建 scanner
        - 批量(record batch)读取 -> 逐行构样本
        - 可选：shuffle_buffer 做大窗口随机打乱
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pairs = self._files_with_fmt  # List[Tuple[Path, str]]
        else:
            wid, nw = worker_info.id, worker_info.num_workers
            pairs = self._files_with_fmt[wid::nw]

        if not pairs:
            # 该 worker 没分到文件，必须返回“可迭代”的空对象
            return iter(())

        pairs = list(pairs)
        random.shuffle(pairs)

        buffer: List[Tuple[Spectrum, float, int, Peptide | List[str]]] = []

        for fpath, fmt in pairs:
            dataset = ds.dataset(str(fpath), format=fmt)
            scanner = dataset.scanner(
                filter=None, use_threads=self.use_threads, batch_size=self.batch_rows
            )

            for rb in scanner.to_batches():
                tbl = pa.Table.from_batches([rb])
                df_batch = _clean_and_remap_batch(tbl)
                if df_batch.height == 0:
                    continue

                for row in df_batch.iter_rows(named=True):
                    try:
                        item = self._build_item_from_row(row)
                    except Exception as e:
                        logger.debug(f"Row failed with {e}; using dummy spectrum.")
                        item = self._dummy_item()

                    if self.shuffle_buffer > 0:
                        buffer.append(item)
                        if len(buffer) >= self.shuffle_buffer:
                            j = random.randrange(len(buffer))
                            yield buffer.pop(j)
                    else:
                        yield item

        while buffer:
            j = random.randrange(len(buffer))
            yield buffer.pop(j)

    # ------------------------ 样本构造与预处理 ------------------------

    def _dummy_item(self) -> Tuple[Spectrum, float, int, Peptide | str]:
        spectrum = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        precursor_mz = 1.0
        precursor_charge = 1
        if self.return_str:
            pep_out: Peptide | List[str] = ""
        else:
            pep_out = torch.zeros(
                (self.peptide_pad_length + (1 if self.add_eos else 0),), dtype=torch.long
            )
        return spectrum, precursor_mz, precursor_charge, pep_out

    def _build_item_from_row(self, row: dict) -> Tuple[Spectrum, float, int, Peptide | str]:
        # 读取/转 dtype（尽量零拷贝或一次拷贝）
        mz_np = np.asarray(row["mz_array"], dtype=np.float32)
        int_np = np.asarray(row["intensity_array"], dtype=np.float32)
        mz = torch.from_numpy(mz_np)
        inten = torch.from_numpy(int_np)

        precursor_mz: float = float(row["precursor_mz"])
        precursor_charge: int = int(row["precursor_charge"])

        peptide_str: str = row["modified_sequence"] if self.annotated else ""

        spectrum = self._process_peaks(mz, inten, precursor_mz, precursor_charge)

        if self.bin_spectra:
            hist = torch.histogram(spectrum[:, 0], bins=self.bins, weight=spectrum[:, 1]).hist
            spectrum = hist  # 1D: (nbins,)
        elif self.pad_spectrum_max_length:
            if spectrum.shape[0] > self.n_peaks:
                spectrum = spectrum[: self.n_peaks]
            if spectrum.shape[0] < self.n_peaks:
                out = torch.zeros((self.n_peaks, 2), dtype=spectrum.dtype)
                out[: spectrum.shape[0]] = spectrum
                spectrum = out

        if self.return_str:
            pep_out: Peptide | List[str] = peptide_str
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

        return spectrum, precursor_mz, precursor_charge, pep_out

    def _process_peaks(
        self,
        mz_array: Tensor,            # shape: (P,)
        int_array: Tensor,           # shape: (P,)
        precursor_mz: float,
        precursor_charge: int,
    ) -> Spectrum:
        """与原版等价的光谱预处理；默认使用 spectrum_utils，可切换快速路径。"""
        if self.use_sus_preprocess:
            mz_np  = mz_array.detach().cpu().numpy().astype(np.float32, copy=False)
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

        # 快速路径：纯张量实现
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

        if int(self.n_peaks) < int_array.numel():
            topk = torch.topk(int_array, k=self.n_peaks, largest=True, sorted=False)
            idx = topk.indices
            mz_array = mz_array[idx]
            int_array = int_array[idx]

        return torch.stack([mz_array, int_array], dim=1).to(torch.float32)

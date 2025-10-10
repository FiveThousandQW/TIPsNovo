from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlsplit

import torch
from jaxtyping import Bool, Float, Integer
from omegaconf import DictConfig
from torch import Tensor, nn

from ..__init__ import console
from ..constants import LEGACY_PTM_TO_UNIMOD, MAX_SEQUENCE_LENGTH
from ..inference import Decodable
from ..transformer.layers import (
    ConvPeakEmbedding,
    MultiScalePeakEmbedding,
    PositionalEncoding,
)
from ..types import (
    DiscretizedMass,
    Peptide,
    PeptideMask,
    PrecursorFeatures,
    ResidueLogits,
    ResidueLogProbabilities,
    Spectrum,
    SpectrumEmbedding,
    SpectrumMask,
)
from ..utils import ResidueSet
from ..utils.colorlogging import ColorLog
from ..utils.file_downloader import download_file

MODEL_TYPE = "transformer"


logger = ColorLog(console, __name__).logger


class InstaNovo(nn.Module, Decodable):
    """The Instanovo model."""

    def __init__(
        self,
        residue_set: ResidueSet,
        dim_model: int = 768,
        n_head: int = 16,
        dim_feedforward: int = 2048,
        n_layers: int = 9,
        dropout: float = 0.1,
        max_charge: int = 5,
        use_flash_attention: bool = False,
        conv_peak_encoder: bool = False,
    ) -> None:
        super().__init__()
        self._residue_set = residue_set
        self.vocab_size = len(residue_set)
        self.use_flash_attention = use_flash_attention
        self.conv_peak_encoder = conv_peak_encoder

        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        if self.use_flash_attention:
            # All input spectra are padded to some max length
            # Pad spectrum replaces zeros in input spectra
            # This is for flash attention (no masks allowed)
            self.pad_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # Encoder
        self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)
        if self.conv_peak_encoder:
            self.conv_encoder = ConvPeakEmbedding(dim_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Decoder
        self.aa_embed = nn.Embedding(self.vocab_size, dim_model, padding_idx=0)

        self.aa_pos_embed = PositionalEncoding(dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # Non-autoregressive decoder components for peptide tag prediction
        tag_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        self.tag_decoder = nn.TransformerDecoder(
            tag_decoder_layer,
            num_layers=max(1, n_layers // 3),
        )
        self.tag_queries = nn.Parameter(torch.randn(MAX_SEQUENCE_LENGTH, dim_model))
        self.tag_pos_embed = PositionalEncoding(dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH)
        self.length_predictor = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, MAX_SEQUENCE_LENGTH),
        )

        self.head = nn.Linear(dim_model, self.vocab_size)
        self.charge_encoder = nn.Embedding(max_charge, dim_model)

    @property
    def residue_set(self) -> ResidueSet:
        """Every model must have a `residue_set` attribute."""
        return self._residue_set

    @staticmethod
    def _get_causal_mask(seq_len: int, return_float: bool = False) -> PeptideMask:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        if return_float:
            return (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
        return ~mask.bool()

    @staticmethod
    def get_pretrained() -> list[str]:
        """Get a list of pretrained model ids."""
        # Load the models.json file
        with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        if MODEL_TYPE not in models_config:
            return []

        return list(models_config[MODEL_TYPE].keys())

    # self 指代 类的实例 (instance)。它用于实例方法中。
    # cls 指代 类本身 (class)。它用于类方法中（用 @classmethod 装饰器标记的方法）。
    @classmethod    # 不需要先创建一个类的实例就能调用这个方法 #
    def load(
        cls, path: str, update_residues_to_unimod: bool = True
    ) -> Tuple["InstaNovo", "DictConfig"]:
        """Load model from checkpoint path."""
        # Add  to allow list
        _whitelist_torch_omegaconf() # 白名单，保护安全
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        config = ckpt["config"] # 加载后得到的 ckpt 是一个字典，通常包含了模型的权重 (state_dict)、训练时的配置 (config) 等信息

        # check if PTL checkpoint
        # 这是一个为了兼容 PyTorch Lightning 框架的巧妙处理。
        # PyTorch Lightning 在保存模型时，会自动给权重字典 (state_dict) 中所有的键（key）加上 "model." 这个前缀。
        # if all(...): 这行代码检查权重字典中是否所有键都以 "model." 开头。
        # 如果是，{k.replace("model.", ""): v ...} 这段字典推导式会遍历所有键值对，并移除每个键的 "model." 前缀。
        # 目的: 这样处理后，这些权重就能被一个普通的、非 Lightning 框架的 PyTorch 模型正确加载。
        if all(x.startswith("model") for x in ckpt["state_dict"].keys()):
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

        # 这段是另一个向后兼容的处理，针对氨基酸残基（residue）和翻译后修饰（PTM）的定义。
        # residues = dict(config["residues"]): 从配置中提取出残基的定义。
        # if update_residues_to_unimod:: 如果函数调用时该参数为 True...
        # {...}: 这段字典推导式会遍历 residues 字典。对于每一个键（代表一个修饰），它会检查这个键是否存在于 LEGACY_PTM_TO_UNIMOD 这个映射字典中。如果存在，就将其替换为新的 UniMod 标准名称。
        # 目的: 这确保了即使用户加载的是一个用旧版代码训练的模型，其修饰定义也能被平滑地升级到当前代码所使用的新标准。
        # residue_set = ResidueSet(residues): 使用最终处理好的 residues 字典，创建一个 ResidueSet 对象。这可能是一个专门管理氨基酸信息的自定义类。
        residues = dict(config["residues"])
        if update_residues_to_unimod:
            residues = {
                LEGACY_PTM_TO_UNIMOD[k] if k in LEGACY_PTM_TO_UNIMOD else k: v
                for k, v in residues.items()
            }
        residue_set = ResidueSet(residues)


        # model = cls(...): 这是类方法的核心体现。它使用 cls（即 InstaNovo 类）来创建一个新的模型实例。
        # 它从 config 对象中读取所有必要的模型架构参数（如 dim_model, n_head 等），确保新创建的这个模型骨架与被保存的模型完全一致。
        # model.load_state_dict(ckpt["state_dict"]): 这是加载过程的最后一步。它调用模型的 load_state_dict 方法，将之前处理好的权重（ckpt["state_dict"]）填充到刚刚创建的模型骨架中。
        model = cls(
            residue_set=residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_charge=config["max_charge"],
            use_flash_attention=config.get("use_flash_attention", False),
            conv_peak_encoder=config.get("conv_peak_encoder", False),
        )
        model.load_state_dict(ckpt["state_dict"])

        return model, config

    @classmethod
    def from_pretrained(
        cls, model_id: str, update_residues_to_unimod: bool = True
    ) -> Tuple["InstaNovo", "DictConfig"]:
        """Download and load by model id or model path."""
        # Check if model_id is a local file path
        if "/" in model_id or "\\" in model_id or model_id.endswith(".ckpt"):
            if os.path.isfile(model_id):
                return cls.load(model_id, update_residues_to_unimod=update_residues_to_unimod)
            else:
                raise FileNotFoundError(f"No file found at path: {model_id}")

        # Load the models.json file
        with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        # Find the model in the config
        if MODEL_TYPE not in models_config or model_id not in models_config[MODEL_TYPE]:
            raise ValueError(
                f"Model {model_id} not found in models.json, options are "
                f"[{', '.join(models_config[MODEL_TYPE].keys())}]"
            )

        model_info = models_config[MODEL_TYPE][model_id]
        url = model_info["remote"]

        # Create cache directory if it doesn't exist
        cache_dir = Path.home() / ".cache" / "instanovo"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate a filename for the cached model
        file_name = urlsplit(url).path.split("/")[-1]
        cached_file = cache_dir / file_name

        # Check if the file is already cached
        if not cached_file.exists():
            download_file(url, cached_file, model_id, file_name)

        else:
            logger.info(f"Model {model_id} already cached at {cached_file}")

        try:
            # Load and return the model
            logger.info(f"Loading model {model_id} (remote)")
            return cls.load(str(cached_file), update_residues_to_unimod=update_residues_to_unimod)
        except Exception as e:
            logger.warning(
                f"Failed to load cached model {model_id}, it may be corrupted. "
                f"Deleting and re-downloading. Error: {e}"
            )
            if cached_file.exists():
                cached_file.unlink()

            download_file(url, cached_file, model_id, file_name)
            logger.info(f"Loading newly downloaded model {model_id}")
            return cls.load(str(cached_file), update_residues_to_unimod=update_residues_to_unimod)

    def forward(
        self,
        x: Float[Spectrum, " batch"],
        p: Float[PrecursorFeatures, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        add_bos: bool = True,
    ) -> Float[ResidueLogits, "batch token+1"]:
        """Model forward pass.

        Args:
            x: Spectra, float Tensor (batch, n_peaks, 2)
            p: Precursors, float Tensor (batch, 3)
            y: Peptide, long Tensor (batch, seq_len, vocab)
            x_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            y_mask: Peptide padding mask, bool Tensor (batch, seq_len)
            add_bos: Force add a <s> prefix to y, bool

        Returns:
            logits: float Tensor (batch, n, vocab_size),
            (batch, n+1, vocab_size) if add_bos==True.
        """
        if self.use_flash_attention:
            x, x_mask = self._flash_encoder(x, p, x_mask)
            return self._flash_decoder(x, y, x_mask, y_mask, add_bos)

        x, x_mask = self._encoder(x, p, x_mask)
        return self._decoder(x, y, x_mask, y_mask, add_bos)

    def init(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[
        Tuple[Float[Spectrum, " batch"], Bool[SpectrumMask, " batch"]],
        Float[ResidueLogProbabilities, "batch token"],
    ]:
        """Initialise model encoder."""
        if self.use_flash_attention:
            spectra, _ = self._encoder(spectra, precursors, None)
            logits = self._decoder(spectra, None, None, None, add_bos=False)
            return (
                spectra,
                torch.zeros(spectra.shape[0], spectra.shape[1]).to(spectra.device),
            ), torch.log_softmax(logits[:, -1, :], -1)

        spectra, spectra_mask = self._encoder(spectra, precursors, spectra_mask)
        logits = self._decoder(spectra, None, spectra_mask, None, add_bos=False)
        return (spectra, spectra_mask), torch.log_softmax(logits[:, -1, :], -1)

    def score_candidates(
        self,
        sequences: Integer[Peptide, " batch"],
        precursor_mass_charge: Float[PrecursorFeatures, " batch"],
        spectra: Float[Spectrum, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Score a set of candidate sequences."""
        if self.use_flash_attention:
            logits = self._flash_decoder(spectra, sequences, None, None, add_bos=True)
        else:
            logits = self._decoder(spectra, sequences, spectra_mask, None, add_bos=True)

        return torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(self, mass_scale: int) -> Integer[DiscretizedMass, " residue"]:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(len(self.residue_set), dtype=torch.int64)
        for index, residue in self.residue_set.index_to_residue.items():
            if residue in self.residue_set.residue_masses:
                residue_masses[index] = round(mass_scale * self.residue_set.get_mass(residue))
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return int(self.residue_set.EOS_INDEX)

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return int(self.residue_set.PAD_INDEX)

    def init_tag_decoder(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[
        Tuple[Float[SpectrumEmbedding, " batch"], Bool[SpectrumMask, " batch"] | None],
        Float[ResidueLogProbabilities, "batch token"],
    ]:
        """Initialise encoder states for non-autoregressive tag decoding."""

        if self.use_flash_attention:
            spectra, _ = self._flash_encoder(spectra, precursors, None)
            length_logits = self.length_predictor(spectra[:, 0, :])
            return (spectra, None), torch.log_softmax(length_logits, dim=-1)

        spectra, spectra_mask = self._encoder(spectra, precursors, spectra_mask)

        if spectra_mask is not None:
            valid = (~spectra_mask).float().unsqueeze(-1)
            pooled = (spectra * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = spectra.mean(dim=1)

        length_logits = self.length_predictor(pooled)
        return (spectra, spectra_mask), torch.log_softmax(length_logits, dim=-1)

    def non_autoregressive_logits(
        self,
        spectra: Float[SpectrumEmbedding, " batch"],
        spectra_mask: Optional[Bool[SpectrumMask, " batch"]],
        length: int,
    ) -> Float[ResidueLogits, " batch"]:
        """Predict logits for a fixed-length peptide tag."""

        if length <= 0:
            raise ValueError("length must be a positive integer")

        queries = self.tag_queries[:length].unsqueeze(0).expand(spectra.shape[0], -1, -1)
        queries = self.tag_pos_embed(queries)

        tag_states = self.tag_decoder(
            queries,
            spectra,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=spectra_mask,
        )

        return self.head(tag_states)

    def decode(self, sequence: Peptide) -> list[str]:
        """Decode a single sequence of AA IDs."""
        # Note: Sequence is reversed as InstaNovo predicts right-to-left.
        # We reverse the sequence again when decoding to ensure
        # the decoder outputs forward sequences.
        return self.residue_set.decode(sequence, reverse=True)  # type: ignore

    def idx_to_aa(self, idx: Peptide) -> list[str]:
        """Decode a single sample of indices to aa list."""
        idx = idx.cpu().numpy()
        t = []
        for i in idx:
            if i == self.eos_id:
                break
            if i == self.bos_id or i == self.pad_id:
                continue
            t.append(i)
        return [self.i2s[x.item()] for x in t]

    def batch_idx_to_aa(self, idx: Integer[Peptide, " batch"], reverse: bool) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.residue_set.decode(i, reverse=reverse) for i in idx]

    def _encoder(
        self,
        x: Float[Spectrum, " batch"],
        p: Float[PrecursorFeatures, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[Float[SpectrumEmbedding, " batch"], Bool[SpectrumMask, " batch"]]:
        if self.conv_peak_encoder:
            x = self.conv_encoder(x)
            x_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device).bool()
        else:
            if x_mask is None:
                x_mask = ~x.sum(dim=2).bool()
            x = self.peak_encoder(x)

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1)
        latent_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([latent_mask, x_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=x_mask)

        # Prepare precursors
        masses = self.peak_encoder.encode_mass(p[:, None, [0]])
        charges = self.charge_encoder(p[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Concatenate precursors
        x = torch.cat([precursors, x], dim=1)
        prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
        x_mask = torch.cat([prec_mask, x_mask], dim=1)

        return x, x_mask

    def _decoder(
        self,
        x: Float[Spectrum, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Bool[SpectrumMask, " batch"],
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        add_bos: bool = True,
    ) -> Float[ResidueLogits, " batch"]:
        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = (
                torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
                * self.residue_set.SOS_INDEX
            )
            y = torch.cat([bos, y], dim=1)

            if y_mask is not None:
                bos_mask = torch.zeros((y_mask.shape[0], 1), dtype=bool, device=y_mask.device)
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = ~y.sum(axis=2).bool()

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y,
            x,
            tgt_mask=c_mask,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )

        return self.head(y_hat)

    def _flash_encoder(self, x: Tensor, p: Tensor, x_mask: Tensor = None) -> tuple[Tensor, Tensor]:
        # Special mask for zero-indices
        # One is padded, zero is normal
        x_mask = (~x.sum(dim=2).bool()).float()


        x = self.peak_encoder(x)
        pad_spectrum = self.pad_spectrum.expand(x.shape[0], x.shape[1], -1)

        # torch.compile doesn't allow dynamic sizes (returned by mask indexing)
        x = x * (1 - x_mask[:, :, None]) + pad_spectrum * (x_mask[:, :, None])

        # Self-attention on latent spectra AND peaks
        latent_spectra = self.latent_spectrum.expand(x.shape[0], -1, -1)
        x = torch.cat([latent_spectra, x], dim=1).contiguous()

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            raise ImportError(
                "Training InstaNovo with Flash attention enabled requires at least pytorch v2.3. "
                "Please upgrade your pytorch version"
            ) from None

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = self.encoder(x)

        # Prepare precursors
        masses = self.peak_encoder.encode_mass(p[:, None, [0]])
        charges = self.charge_encoder(p[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Concatenate precursors
        x = torch.cat([precursors, x], dim=1).contiguous()

        return x, None

    def _flash_decoder(
        self,
        x: Tensor,
        y: Tensor,
        x_mask: Tensor,
        y_mask: Tensor = None,
        add_bos: bool = True,
    ) -> Tensor:
        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = (
                torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device)
                * self.residue_set.SOS_INDEX
            )
            y = torch.cat([bos, y], dim=1)

        y = self.aa_embed(y)

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            raise ImportError(
                "Training InstaNovo with Flash attention enabled requires at least pytorch v2.3. "
                "Please upgrade your pytorch version"
            ) from None

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y_hat = self.decoder(y, x, tgt_mask=c_mask)

        return self.head(y_hat)


def _whitelist_torch_omegaconf() -> None:
    """Whitelist specific modules for loading configs from checkpoints."""
    # This is done to safeguard against arbitrary code execution from checkpoints.
    from collections import defaultdict
    from typing import Any

    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ContainerMetadata,
            Metadata,
            ListConfig,
            AnyNode,
            Any,  # Only used for type hinting in omegaconf.
            defaultdict,
            dict,
            list,
            int,
        ]
    )

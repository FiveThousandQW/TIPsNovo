"""Non-autoregressive decoding utilities for peptide tag generation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
from jaxtyping import Bool, Float

from .interfaces import Decoder, Decodable, ScoredSequence
from ..types import PrecursorFeatures, Spectrum, SpectrumMask


@dataclass
class TagCandidate(ScoredSequence):
    """Representation of a predicted peptide tag."""

    length: int
    length_log_probability: float


class NonAutoregressiveTagDecoder(Decoder):
    """Generate high-quality peptide tags using non-autoregressive decoding."""

    def __init__(
        self,
        model: Decodable,
        *,
        min_length: int = 4,
        max_length: int = 12,
        top_k: int = 5,
        per_length_beam: int | None = None,
        allow_unknown: bool = True,
        unknown_token: str = "X",
        unknown_probability_threshold: float = 0.4,
    ) -> None:
        super().__init__(model)
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.min_length = min_length
        self.max_length = max_length
        self.top_k = top_k
        self.per_length_beam = per_length_beam or top_k
        self.allow_unknown = allow_unknown
        self.unknown_token = unknown_token
        self.unknown_probability_threshold = unknown_probability_threshold

        self._invalid_indices: Sequence[int] = (
            int(self.model.get_empty_index()),
            int(self.model.residue_set.SOS_INDEX),
            int(self.model.get_eos_index()),
        )

    def decode(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"] | None = None,
    ) -> list[list[TagCandidate]]:
        """Predict peptide tags for a batch of spectra."""

        (encoded, encoded_mask), length_log_probs = self.model.init_tag_decoder(
            spectra, precursors, spectra_mask
        )

        batch_results: list[list[TagCandidate]] = []
        for index in range(spectra.shape[0]):
            encoded_slice = encoded[index : index + 1]
            mask_slice = None if encoded_mask is None else encoded_mask[index : index + 1]
            batch_results.append(
                self._decode_single(
                    encoded_slice,
                    mask_slice,
                    length_log_probs[index],
                )
            )

        return batch_results

    def _decode_single(
        self,
        encoded: Float[Spectrum, " batch"],
        encoded_mask: Bool[SpectrumMask, " batch"] | None,
        length_log_probs: torch.Tensor,
    ) -> list[TagCandidate]:
        candidates: list[TagCandidate] = []

        tag_query_limit = getattr(self.model, "tag_queries").shape[0]
        max_length = min(self.max_length, tag_query_limit, length_log_probs.numel())
        for length in range(self.min_length, max_length + 1):
            logits = self.model.non_autoregressive_logits(encoded, encoded_mask, length)
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
            log_probs[:, list(self._invalid_indices)] = float("-inf")

            beam_sequences = self._beam_search(log_probs, length)
            length_log_prob = float(length_log_probs[length - 1])

            for indices, seq_log_prob, token_logs in beam_sequences:
                residues = self._indices_to_residues(indices, token_logs)
                total_log_prob = seq_log_prob + length_log_prob
                candidates.append(
                    TagCandidate(
                        sequence=residues,
                        mass_error=0.0,
                        sequence_log_probability=total_log_prob,
                        token_log_probabilities=[float(x) for x in token_logs],
                        length=length,
                        length_log_probability=length_log_prob,
                    )
                )

        candidates.sort(key=lambda item: item.sequence_log_probability, reverse=True)
        return candidates[: self.top_k]

    def _beam_search(
        self,
        log_probs: torch.Tensor,
        length: int,
    ) -> List[tuple[list[int], float, list[float]]]:
        sequences: List[tuple[list[int], float, list[float]]] = [([], 0.0, [])]

        for position in range(length):
            step_probs = log_probs[position]
            top_values, top_indices = torch.topk(step_probs, self.per_length_beam)
            next_sequences: List[tuple[list[int], float, list[float]]] = []

            for seq, seq_log_prob, token_logs in sequences:
                for value, index in zip(top_values.tolist(), top_indices.tolist()):
                    if value == float("-inf"):
                        continue
                    next_sequences.append(
                        (
                            seq + [index],
                            seq_log_prob + value,
                            token_logs + [value],
                        )
                    )

            if not next_sequences:
                break

            next_sequences.sort(key=lambda item: item[1], reverse=True)
            sequences = next_sequences[: self.per_length_beam]

        completed = [seq for seq in sequences if len(seq[0]) == length]
        return completed

    def _indices_to_residues(
        self,
        indices: Sequence[int],
        token_log_probs: Sequence[float],
    ) -> list[str]:
        residues: list[str] = []
        for idx, log_prob in zip(indices, token_log_probs):
            probability = math.exp(float(log_prob))
            if (
                self.allow_unknown
                and probability < self.unknown_probability_threshold
            ):
                residues.append(self.unknown_token)
            else:
                residues.append(self.model.residue_set.index_to_residue[int(idx)])
        return residues

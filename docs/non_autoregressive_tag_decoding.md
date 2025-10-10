# Non-autoregressive peptide tag decoding additions

This document provides a detailed walkthrough of the architectural and API
changes that were introduced to extend InstaNovo with non-autoregressive peptide
*tag* generation capabilities. The goal of the update is to emit short, high
confidence sequence fragments ("tags") that can be used as BLAST seeds, instead
of forcing the model to always predict a full-length peptide in a purely
autoregressive manner.

## 1. New inference entry points (`inference/__init__.py`)

We export two new helper types so that downstream scripts can instantiate the
non-autoregressive decoder directly:

- `NonAutoregressiveTagDecoder` – orchestration logic that calls into the model
  to compute tag logits and performs beam-search style hypothesis pruning.
- `TagCandidate` – a dataclass that extends the existing `ScoredSequence`
  payload with metadata that is useful for tag workflows (predicted tag length
  and its probability).

Adding them to `__all__` makes the classes available through
`from instanovo.inference import NonAutoregressiveTagDecoder` without touching
internal paths.

## 2. Model-side extensions (`transformer/model.py`)

To support non-autoregressive predictions the `InstaNovo` model now owns a small
set of additional submodules that are only used in tag decoding mode:

1. **Tag decoder stack.** A lightweight Transformer decoder (`self.tag_decoder`)
   consumes a bank of learned queries instead of the usual autoregressive
   teacher-forced tokens. We reuse the existing dimensionality, attention head
   count, and feed-forward width so that the encoder activations remain
   compatible. The number of layers defaults to one third of the full decoder
   depth to keep the extra compute modest.

2. **Learned tag queries.** `self.tag_queries` is a parameter matrix shaped like
   `[MAX_SEQUENCE_LENGTH, dim_model]`. During inference we slice the first *L*
   rows (where *L* is the requested tag length) to obtain a set of query vectors
   that attend to the encoded spectrum in parallel. Positional encodings are
   injected via the shared `PositionalEncoding` module so each position learns a
   distinct receptive field.

3. **Length predictor.** A small two-layer MLP converts a pooled encoder state
   into a categorical distribution over possible tag lengths. This allows the
   decoder to rank tags of heterogeneous lengths within the same beam.

In addition to the new submodules, two helper methods are exposed:

- `init_tag_decoder(...)` runs the spectrum encoder and the length predictor.
  It returns the encoder memory/mask pair needed by the decoder and the log
  probabilities over tag lengths. Flash-attention and standard attention flows
  are both supported.
- `non_autoregressive_logits(...)` applies the tag decoder to a fixed target
  length, yielding per-position residue logits that are later transformed into
  log probabilities.

These methods complement the existing autoregressive utilities and are invoked
by the new inference helper described next.

## 3. Inference helper (`inference/tag_decoder.py`)

The `NonAutoregressiveTagDecoder` orchestrates the full workflow:

1. **Batch orchestration.** `decode(...)` calls
   `model.init_tag_decoder(...)` once per spectrum batch to obtain encoder
   activations and length distributions. Each spectrum is then decoded
   independently to simplify bookkeeping.

2. **Per-length beam search.** For each allowed tag length we query
   `model.non_autoregressive_logits(...)` to obtain position-wise log
   probabilities. A beam search keeps the top `per_length_beam` sequences at
   every step, ensuring that only high-quality hypotheses survive. Residues that
   correspond to invalid tokens (PAD, SOS, EOS) are masked out by setting their
   log-probabilities to `-inf`.

3. **Unknown residue marking.** While constructing the final amino-acid strings
   we optionally replace low-confidence residues with a configurable unknown
   token (default `X`). The decision is driven by the per-position probability
   mass—once it drops below `unknown_probability_threshold`, the residue is
   considered unreliable and replaced.

4. **Result packaging.** Each surviving sequence is turned into a
   `TagCandidate`, which records the amino-acid list, token-wise log
   probabilities, the aggregated sequence log-probability, the chosen tag length
   and the corresponding length log-probability. Candidates are finally sorted
   by total log-probability and the top-`k` items are returned per spectrum.

The decoder exposes several configuration knobs that mirror the user goals:

- `min_length` / `max_length` constrain the tag lengths considered.
- `top_k` controls how many tags are surfaced per spectrum.
- `per_length_beam` adjusts the beam width for each length (defaults to `top_k`).
- `allow_unknown`, `unknown_token`, and
  `unknown_probability_threshold` govern the handling of low-confidence
  positions.

Together these components allow InstaNovo to emit shorter, high-quality peptide
fragments that can be fed to downstream search heuristics such as `blastp` while
still reusing the bulk of the trained model.

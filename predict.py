from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from __init__ import console
from constants import ANNOTATED_COLUMN, ANNOTATION_ERROR, MASS_SCALE, MAX_MASS
from inference import (
    BeamSearchDecoder,
    Decoder,
    GreedyDecoder,
    Knapsack,
    KnapsackBeamSearchDecoder,
    ScoredSequence,
)
from transformer.dataset import SpectrumDataset, collate_batch
from transformer.model import InstaNovo
from utils import Metrics, SpectrumDataFrame
from utils.colorlogging import ColorLog
from utils.device_handler import check_device
from utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference"


def get_preds(
    config: DictConfig,
    model: InstaNovo,
    model_config: DictConfig,
    s3: S3FileHandler,
) -> None:
    # 检查参数和是否有输出的位置
    """Get predictions from a trained model."""
    if config.get("denovo", False) and config.get("output_path", None) is None:
        raise ValueError(
            "Must specify an output csv path in denovo mode. Please specify in config "
            "or with the cli flag --output-path `path/to/output.csv`"
        )

    data_path = config["data_path"]

    # 如果 data_path 是一个列表，代码会遍历它，为每个数据集（组）建立文件路径到组名的映射 (group_mapping) 和组名到输出路径的映射
    if OmegaConf.is_list(data_path):
        _new_data_paths = []
        group_mapping = {}
        group_output = {}
        for group in data_path:
            path = group.get("input_path")
            name = group.get("result_name")
            for fp in SpectrumDataFrame._convert_file_paths(path):
                group_mapping[fp] = name
            _new_data_paths.append(path)
            group_output[name] = group.get("output_path")
        data_path = _new_data_paths
    else:
        group_mapping = None
        group_output = None

    output_path = config.get("output_path", None)

    # Some commomly used config variables
    denovo = config.get("denovo", False)
    num_beams = config.get("num_beams", 1)
    use_basic_logging = config.get("use_basic_logging", True)
    save_beams = config.get("save_beams", False)
    device = check_device(config=config)
    logger.info(f"Using device: {device} for InstaNovo predictions")
    fp16 = config.get("fp16", True)

    if fp16 and device.lower() == "cpu":
        logger.warning("fp16 is enabled but device type is cpu. fp16 will be disabled.")
        fp16 = False

    logger.info(f"Loading data from {data_path}")


    try:
        # 加载谱图数据
        sdf = SpectrumDataFrame.load(
            data_path,
            lazy=False,
            is_annotated=not denovo,
            column_mapping=config.get("column_map", None),
            shuffle=False,
            add_spectrum_id=True,
            add_source_file_column=True,
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        # 如果数据中缺少肽段序列注释（在评估模式下是必需的），它会给出一个更友好的错误提示，建议用户使用 denovo=True 标志
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, "
                "are you trying to run de novo prediction? Add the `denovo=True` flag"
            ) from e
        else:
            raise

    # Check max charge values:
    # 代码会根据模型支持的最大电荷数 (max_charge) 过滤掉不合格的谱图
    original_size = len(sdf)
    max_charge = config.get("max_charge", 10)
    model_max_charge = model_config.get("max_charge", 10)
    if max_charge > model_max_charge:
        logger.warning(
            f"Inference has been configured with max_charge={max_charge}, "
            f"but model has max_charge={model_max_charge}."
        )
        logger.warning(f"Overwriting max_charge config to model value: {model_max_charge}.")
        max_charge = model_max_charge

    sdf.filter_rows(
        lambda row: (row["precursor_charge"] <= max_charge) and (row["precursor_charge"] > 0)
    )
    if len(sdf) < original_size:
        logger.warning(
            f"Found {original_size - len(sdf)} rows with charge > {max_charge}. "
            "These rows will be skipped."
        )

    # 允许用户只对数据的一个子集进行预测，方便快速测试
    subset = config.get("subset", 1.0)
    if not 0 < subset <= 1:
        raise ValueError(
            f"Invalid subset value: {subset}. Must be a float greater than 0 and less than or equal to 1."  # noqa: E501
        )

    sdf.sample_subset(fraction=subset, seed=42)
    logger.info(f"Data loaded, evaluating {subset * 100:.2f}%, {len(sdf):,} samples in total.")

    # 检查加载后是否还有数据，如果没有则退出。
    assert sdf.df is not None
    if sdf.df.is_empty():
        logger.warning("No data found, exiting.")
        sys.exit()

    # residue_set = model.residue_set: 获取模型能理解的氨基酸和修饰的“词汇表”。
    # residue_set.update_remapping(...): 允许用户动态地提供一个映射，将数据中的修饰写法（如 "M(ox)"）映射到模型内部的表示（如 "M[UNIMOD:35]"）。
    residue_set = model.residue_set
    logger.info(f"Vocab: {residue_set.index_to_residue}")
    residue_set.update_remapping(config.get("residue_remapping", {}))


    # 在评估模式下，代码会检查真实标签（target peptides）中是否含有模型词汇表之外的氨基酸。如果有，会警告用户并过滤掉这些数据行。
    if not denovo:
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        data_residues = sdf.get_vocabulary(residue_set.tokenize)
        if len(data_residues - supported_residues) > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            logger.warning(f"Residues found: \n{data_residues - supported_residues}")
            logger.warning(
                "Please check residue remapping if a different convention has been used."
            )
            original_size = len(sdf)
            sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            logger.warning(f"{original_size - len(sdf):,d} rows have been dropped.")
            logger.warning("Peptide recall should recalculated accordingly.")


    # Used to group validation outputs
    if group_mapping is not None:
        logger.info("Computing validation groups.")
        sequence_groups = pd.Series(
            [
                group_mapping[row["source_file"]]
                if row.get("source_file", None) in group_mapping
                else "no_group"
                for row in iter(sdf)
            ]
        )
        logger.info("Sequences per validation group:")
        for group in sequence_groups.unique():
            logger.info(f" - {group}: {(sequence_groups == group).sum():,d}")
    else:
        sequence_groups = None

    # 将加载好的数据 (sdf) 封装成一个 PyTorch Dataset 对象。Dataset 的任务是将每一条数据（一个谱图）转换成模型可以接受的张量（Tensor）格式
    ds = SpectrumDataset(
        sdf,
        residue_set,
        model_config.get("n_peaks", 200),
        return_str=True,
        annotated=not denovo,
    )

    # 将 Dataset 对象封装成一个 DataLoader。DataLoader 负责将数据分批 (batching)，这对于高效地在 GPU 上进行计算至关重要。
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        num_workers=0,  # sdf requirement, handled internally
        shuffle=False,  # sdf requirement, handled internally
        collate_fn=collate_batch,
    )

    model = model.to(device)
    # 一个至关重要的步骤。它将模型切换到评估模式，这会关闭 Dropout 等只在训练时使用的层，确保预测结果是确定的。
    model = model.eval()

    # Setup decoder

    # 选择和启动decoder
    # KnapsackBeamSearchDecoder: 背包波束搜索。一种高级的解码方法，它能确保最终预测出的肽段序列的理论质量与测量的母离子质量精确匹配。
    # BeamSearchDecoder: 普通波束搜索。在每一步都保留 num_beams 个最可能的候选序列，比贪心搜索更可能找到最优解。
    # GreedyDecoder: 贪心搜索。在每一步都只选择当前最可能的氨基酸，速度最快但准确性可能稍低。
    if config.get("use_knapsack", False):
        logger.info(f"Using Knapsack Beam Search with {num_beams} beam(s)")
        knapsack_path = config.get("knapsack_path", None)
        if knapsack_path is None or not os.path.exists(knapsack_path):
            logger.info("Knapsack path missing or not specified, generating...")
            knapsack = _setup_knapsack(model)
            decoder: Decoder = KnapsackBeamSearchDecoder(model, knapsack)
            if knapsack_path is not None:
                logger.info(f"Saving knapsack to {knapsack_path}")
                knapsack.save(knapsack_path)
        else:
            logger.info("Knapsack path found. Loading...")
            decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)
    elif num_beams > 1:
        logger.info(f"Using Beam Search with {num_beams} beam(s)")
        decoder = BeamSearchDecoder(model=model)
    else:
        logger.info(f"Using Greedy Search with  {num_beams} beam(s)")
        decoder = GreedyDecoder(
            model=model,
            suppressed_residues=config.get("suppressed_residues", None),
            disable_terminal_residues_anywhere=config.get(
                "disable_terminal_residues_anywhere", True
            ),
        )

    # # ... (初始化用于存储结果的列表和字典) ...
    index_cols = config.get("index_columns", ["precursor_mz", "precursor_charge"])
    cols = [x for x in sdf.df.columns if x in index_cols]

    pred_df = sdf.df.to_pandas()[cols].copy()

    preds: dict[int, list[list[str]]] = {i: [] for i in range(num_beams)}
    targs: list[str] = []
    sequence_log_probs: dict[int, list[float]] = {i: [] for i in range(num_beams)}
    token_log_probs: dict[int, list[list[float]]] = {i: [] for i in range(num_beams)}


    start = time.time()

    iter_dl: enumerate[Any] | tqdm[tuple[int, Any]] = enumerate(dl)
    if not use_basic_logging:
        iter_dl = tqdm(enumerate(dl), total=len(dl)) # 设置进度条

    logger.info("Starting evaluation...")

    # 对于每一个批次进预测和调用
    for i, batch in iter_dl:
        spectra, precursors, spectra_mask, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)
        spectra_mask = spectra_mask.to(device)

        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16),
        ):
            batch_predictions = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=num_beams,
                max_length=config.get("max_length", 40),
                return_beam=save_beams,
            )

        #  判断是否处于“保存所有波束（beam）”模式，还是只保存最优的结果
        if save_beams:
            batch_predictions = cast(list[list[ScoredSequence]], batch_predictions)
            for predictions in batch_predictions:
                for j in range(num_beams):
                    if j >= len(predictions):
                        preds[j].append([])
                        sequence_log_probs[j].append(-float("inf"))
                        token_log_probs[j].append([])
                    else:
                        preds[j].append(predictions[j].sequence)
                        sequence_log_probs[j].append(predictions[j].sequence_log_probability)
                        token_log_probs[j].append(predictions[j].token_log_probabilities)
        else:
            batch_predictions = cast(list[ScoredSequence], batch_predictions)
            for prediction in batch_predictions:
                if isinstance(prediction, ScoredSequence):
                    preds[0].append(prediction.sequence)
                    sequence_log_probs[0].append(prediction.sequence_log_probability)
                    token_log_probs[0].append(prediction.token_log_probabilities)
                else:
                    preds[0].append([])
                    sequence_log_probs[0].append(-float("inf"))
                    token_log_probs[0].append([])

        # 将当前批次的真实肽段序列（peptides）追加到 targs 列表中。这用于后续在评估模式下与预测结果进行比较。
        targs += list(peptides)

        # 在循环内部，每隔一定数量的批次（由 log_interval 配置）或者在最后一个批次完成时，打印一次当前的进度，包括已用时间、预计剩余时间和平均每批次耗时。
        # delta = time.time() - start: 在循环结束后，计算整个预测过程的总耗时
        if use_basic_logging and (
            (i + 1) % config.get("log_interval", 50) == 0 or (i + 1) == len(dl)
        ):
            delta = time.time() - start
            est_total = delta / (i + 1) * (len(dl) - i - 1)
            logger.info(
                f"Batch {i + 1:05d}/{len(dl):05d}, [{_format_time(delta)}/"
                f"{_format_time(est_total)}, {(delta / (i + 1)):.3f}s/it]"
            )
    delta = time.time() - start


    logger.info(f"Time taken for {data_path} is {delta:.1f} seconds")
    if len(dl) > 0:
        logger.info(
            f"Average time per batch (bs={config['batch_size']}): {delta / len(dl):.1f} seconds"
        )

    # 这部分代码将之前在循环中收集的所有列表数据，整理并添加为 pred_df 这个 pandas DataFrame 的列。
    # if not denovo:: 在评估模式下，添加一个名为 targets 的列，包含真实的肽段序列。
    # pred_df[...] = ...: 创建多个新列：
    # predictions: 最优预测的肽段序列（由氨基酸列表 join 成字符串）。
    # predictions_tokenised: 逗号分隔的氨基酸序列。
    # if save_beams:: 如果保存了所有波束，则用一个循环为每个波束创建一个新的列（如 preds_beam_1, log_probs_beam_1 等）。
    if not denovo:
        pred_df["targets"] = targs
    pred_df[config.get("pred_col", "predictions")] = ["".join(x) for x in preds[0]]
    pred_df[config.get("pred_tok_col", "predictions_tokenised")] = [", ".join(x) for x in preds[0]]
    pred_df[config.get("log_probs_col", "log_probabilities")] = sequence_log_probs[0]
    pred_df[config.get("token_log_probs_col", "token_log_probabilities")] = token_log_probs[0]

    if save_beams:
        for i in range(num_beams):
            pred_df[f"preds_beam_{i}"] = ["".join(x) for x in preds[i]]
            pred_df[f"log_probs_beam_{i}"] = sequence_log_probs[i]
            pred_df[f"token_log_probs_{i}"] = token_log_probs[i]

    # Always calculate delta_mass_ppm, even in de novo mode
    # 初始化一个 Metrics 辅助类的实例，用于后续的各种性能指标计算。
    metrics = Metrics(residue_set, config.get("isotope_error_range", [0, 1]))

    # 这个匿名函数会计算当前行预测出的肽段序列的理论质量，与测量的母离子质量进行比较，得出它们之间的质量误差（单位是百万分率, ppm）。这是一个衡量预测质量的重要指标。
    # Calculate some additional information for filtering:
    pred_df["delta_mass_ppm"] = pred_df.apply(
        lambda row: np.min(
            np.abs(
                metrics.matches_precursor(
                    preds[0][row.name], row["precursor_mz"], row["precursor_charge"]
                )[1]
            )
        ),
        axis=1,
    )

    # Calculate metrics
    if not denovo:
        # Make sure we pass preds[0] without joining on ""
        # This is to handle cases where n-terminus modifications could be accidentally joined
        # # ... (计算并打印总体性能) ...
        aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
            pred_df["targets"], preds[0]
        )
        aa_er = metrics.compute_aa_er(pred_df["targets"], preds[0])
        auc = metrics.calc_auc(
            pred_df["targets"],
            preds[0],
            np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
        )

        logger.info(f"Performance on {data_path}:")
        logger.info(f"  aa_er       {aa_er:.5f}")
        logger.info(f"  aa_prec     {aa_prec:.5f}")
        logger.info(f"  aa_recall   {aa_recall:.5f}")
        logger.info(f"  pep_prec    {pep_prec:.5f}")
        logger.info(f"  pep_recall  {pep_recall:.5f}")
        logger.info(f"  auc         {auc:.5f}")

        #
        fdr = config.get("filter_fdr_threshold", None)
        if fdr:
            # # ... (计算并打印在特定FDR下的性能) ...
            _, threshold = metrics.find_recall_at_fdr(
                pred_df["targets"],
                preds[0],
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
                fdr=fdr,
            )
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                pred_df["targets"],
                preds[0],
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")]),
                threshold=threshold,
            )
            logger.info(f"Performance at {fdr * 100:.1f}% FDR:")
            logger.info(f"  aa_prec     {aa_prec:.5f}")
            logger.info(f"  aa_recall   {aa_recall:.5f}")
            logger.info(f"  pep_prec    {pep_prec:.5f}")
            logger.info(f"  pep_recall  {pep_recall:.5f}")
            logger.info(f"  confidence  {threshold:.5f}")

        filter_precursor_ppm = config.get("filter_precursor_ppm", None)
        if filter_precursor_ppm:
            # # ... (计算并打印在特定质量误差下的性能) ...
            idx = pred_df["delta_mass_ppm"] < filter_precursor_ppm
            logger.info(f"Performance with filtering at {filter_precursor_ppm} ppm delta mass:")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(preds[0])
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                    pred_df["targets"], filtered_preds
                )
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(
                    f"Rows filtered: {len(sdf) - np.sum(idx)} "
                    f"({(len(sdf) - np.sum(idx)) / len(sdf) * 100:.2f}%)"
                )
                if np.sum(idx) < 1000:
                    logger.info(
                        f"Metrics calculated on a small number of samples ({np.sum(idx)}), "
                        "interpret with care!"
                    )
            else:
                logger.info("No predictions met criteria, skipping metrics.")

        model_confidence_no_pred = config.get("filter_confidence", None)
        if model_confidence_no_pred:
            idx = (
                np.exp(pred_df[config.get("log_probs_col", "log_probabilities")])
                > model_confidence_no_pred
            )
            logger.info(f"Performance with filtering confidence < {model_confidence_no_pred}")
            if np.sum(idx) > 0:
                filtered_preds = pd.Series(preds[0])
                filtered_preds[~idx] = ""
                aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                    pred_df["targets"], filtered_preds
                )
                logger.info(f"  aa_prec     {aa_prec:.5f}")
                logger.info(f"  aa_recall   {aa_recall:.5f}")
                logger.info(f"  pep_prec    {pep_prec:.5f}")
                logger.info(f"  pep_recall  {pep_recall:.5f}")
                logger.info(
                    f"Rows filtered: {len(sdf) - np.sum(idx)} "
                    f"({(len(sdf) - np.sum(idx)) / len(sdf) * 100:.2f}%)"
                )
                if np.sum(idx) < 1000:
                    logger.info(
                        f"Metrics calculated on a small number of samples ({np.sum(idx)}), "
                        "interpret with care!"
                    )
            else:
                logger.info("No predictions met criteria, skipping metrics.")

    # Evaluate individual result files
    # .if sequence_groups is not None ...: 如果输入是多个数据集，代码会：
    # 遍历每个组，单独计算其性能指标。
    # 将这些组的性能指标追加到一个总的摘要结果文件中。
    # 为每个组单独保存一个包含其所有详细预测的 CSV 文件。
    # if output_path is not None:: 将包含所有谱图（无论是否分组）的完整预测结果 pred_df 保存到用户指定的最终输出文件中。
    # s3.upload_to_s3_wrapper(...): 如果配置了 S3，这个辅助函数会将保存的文件上传到云端。

    if sequence_groups is not None and not denovo:
        _preds = pd.Series(preds[0])
        _targs = pd.Series(pred_df["targets"])
        _probs = pd.Series(pred_df[config.get("log_probs_col", "log_probabilities")])

        results = {
            "run_name": config.get("run_name"),
            "instanovo_model": config.get("instanovo_model"),
            "num_beams": num_beams,
            "use_knapsack": config.get("use_knapsack", False),
        }
        for group in sequence_groups.unique():
            if group == "no_group":
                continue
            idx = sequence_groups == group
            _group_preds = _preds[idx].reset_index(drop=True)
            _group_targs = _targs[idx].reset_index(drop=True)
            _group_probs = _probs[idx].reset_index(drop=True)
            aa_prec, aa_recall, pep_recall, pep_prec = metrics.compute_precision_recall(
                _group_targs, _group_preds
            )
            aa_er = metrics.compute_aa_er(_group_targs, _group_preds)
            auc = metrics.calc_auc(_group_targs, _group_preds, _group_probs)

            results.update(
                {
                    f"{group}_aa_prec": [aa_prec],
                    f"{group}_aa_recall": [aa_recall],
                    f"{group}_pep_recall": [pep_recall],
                    f"{group}_pep_prec": [pep_prec],
                    f"{group}_aa_er": [aa_er],
                    f"{group}_auc": [auc],
                }
            )

            fdr = config.get("filter_fdr_threshold", None)
            if fdr:
                _, threshold = metrics.find_recall_at_fdr(
                    _group_targs, _group_preds, np.exp(_group_probs), fdr=fdr
                )
                _, _, pep_recall_at_fdr, _ = metrics.compute_precision_recall(
                    _group_targs,
                    _group_preds,
                    np.exp(_group_probs),
                    threshold=threshold,
                )

                results.update(
                    {
                        f"{group}_pep_recall_at_{fdr:.3f}_fdr": [pep_recall_at_fdr],
                    }
                )

        result_path = config.get("result_file_path")
        local_path = s3.get_local_path(result_path, missing_ok=True)
        if local_path is not None and os.path.exists(local_path):
            results_df = pd.read_csv(local_path)
            results_df = pd.concat(
                [results_df, pd.DataFrame(results)], ignore_index=True, join="outer"
            )
        else:
            results_df = pd.DataFrame(results)

        s3.upload_to_s3_wrapper(results_df.to_csv, config.get("result_file_path"), index=False)

    # Save individual result files per group
    if sequence_groups is not None and group_output is not None:
        for group in sequence_groups.unique():
            idx = sequence_groups == group
            if group_output[group] is not None:
                s3.upload_to_s3_wrapper(pred_df[idx].to_csv, group_output[group], index=False)

    # Save output
    if output_path is not None:
        s3.upload_to_s3_wrapper(pred_df.to_csv, output_path, index=False)
        # pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Upload to Aichor
        if S3FileHandler._aichor_enabled() and not output_path.startswith("s3://"):
            s3.upload(output_path, S3FileHandler.convert_to_s3_output(output_path))


# 是为“背包波束搜索”（Knapsack Beam Search）解码算法准备和配置其所需的数据结构
def _setup_knapsack(model: InstaNovo) -> Knapsack:
    residue_masses = dict(model.residue_set.residue_masses.copy())
    negative_residues = [k for k, v in residue_masses.items() if v < 0]
    if len(negative_residues) > 0:
        logger.warning(f"Negative mass found in residues: {negative_residues}.")
        logger.warning(
            "These residues will be disabled when using knapsack decoding. "
            "A future release is planned to support negative masses."
        )
        residue_masses.update(dict.fromkeys(negative_residues, MAX_MASS))
    for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
        residue_masses[special_residue] = 0
    residue_indices = model.residue_set.residue_to_index
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=MAX_MASS,
        mass_scale=MASS_SCALE,
    )


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

from multiprocessing.pool import worker
from pathlib import Path
import polars as pl
import numpy as np
import re,os
import subprocess
import random,time
from multiprocessing import Pool
import pymzml
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyteomics import mzml
from tqdm import tqdm
import time
from itertools import islice


def get_single_mzml_precision(mzml_path: str) -> tuple[str, str]:
    """
    工作函数：处理单个mzML文件，返回其精度类型。
    优先判断低精度：只要在SCAN_LIMIT范围内发现一个低精度谱图，整个文件即被标记。

    Args:
        mzml_path (str): 单个mzML文件的路径。

    Returns:
        tuple[str, str]: 一个元组，包含(文件路径, 精度类型)。
                         精度类型为 "High_Resolution", "Low_Resolution", 或 "Unknown"。
    """
    # 关键词列表保持不变
    FTMS_KEYWORDS = ['orbitrap', 'fourier transform', 'ftms']
    ITMS_KEYWORDS = ['ion trap', 'itms']
    TOF_KEYWORDS = ['time-of-flight', 'tof']  # TOF 通常是高精度，但这里我们主要区分 FTMS 和 ITMS

    # 新增一个可配置的扫描上限
    # 检查前100个谱图，足以在绝大多数情况下判断是否存在低精度的MS2谱图
    SCAN_LIMIT = 100

    try:
        with mzml.MzML(mzml_path, iterparse=True) as reader:
            # 标记位，用于记录是否在扫描范围内发现过FTMS关键词
            found_ftms_keyword = False
            spectra_checked = 0

            # 使用 islice 高效地只迭代前 SCAN_LIMIT 个谱图
            for spectrum in islice(reader, SCAN_LIMIT):
                spectra_checked += 1
                metadata_text = str(spectrum.keys()).lower() + str(spectrum.values()).lower()

                # 核心逻辑：优先判断低精度
                # 只要发现任何一个谱图是离子阱产生的，立即判定为低精度并返回
                if any(keyword in metadata_text for keyword in ITMS_KEYWORDS):
                    return (mzml_path, "Low_Resolution")

                # 如果没发现低精度，再检查是否含有高精度关键词
                if not found_ftms_keyword and any(keyword in metadata_text for keyword in FTMS_KEYWORDS):
                    found_ftms_keyword = True

            # -- 循环结束后的判断 --

            # 1. 如果循环正常结束（即在前 SCAN_LIMIT 个谱图中都未找到ITMS关键词）
            if spectra_checked == 0:
                return (mzml_path, "Empty_or_Error")

            # 2. 如果在扫描范围内发现了FTMS关键词，且没有发现ITMS，则判定为高精度
            if found_ftms_keyword:
                return (mzml_path, "High_Resolution")

            # 3. 如果什么关键词都没发现
            return (mzml_path, "Unknown")

    except FileNotFoundError:
        return (mzml_path, "File_Not_Found")
    except Exception:
        return (mzml_path, "Parsing_Error")


def get_precision_parallel(mzml_paths: list[str], num_workers: int = 16) -> dict[str, str]:
    """
    主函数：使用并行处理高效地获取大量mzML文件的精度类型。

    Args:
        mzml_paths (list[str]): 包含多个mzML文件路径的列表。
        num_workers (int): 使用的工作进程数。

    Returns:
        dict[str, str]: 一个字典，键是mzML文件路径，值是其精度类型。
    """
    print(f"启动 {num_workers} 个工作进程来处理 {len(mzml_paths)} 个文件...")
    results_dict = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(get_single_mzml_precision, path): path for path in mzml_paths}

        for future in tqdm(as_completed(futures), total=len(mzml_paths), desc="Processing mzML files"):
            try:
                path, precision_type = future.result()
                results_dict[path] = precision_type
            except Exception as e:
                path = futures[future]
                results_dict[path] = f"Critical_Error: {e}"

    return results_dict


def SysteMHC_Filter():
    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Allele_peptide_HLA.csv')
    df = df.drop_nulls(subset=['Allele_Count'])
    # binder 共有4093W PSM
    # PTM_list = ['','M[147]','P[113]','W[202]','E[111]','Q[111]']
    PTM_list = ['', 'M[147]']
    df_filtered = df.filter(
        pl.col('Modification_Site').is_in(PTM_list)
    )
    # PSM 5098w

    tolerance_dic = np.load('/data/qwu/data/TIPnovo/systemMHC/System_analyzer_dic.npy', allow_pickle=True)[()]
    all_mzML_files_dic = np.load('/data/qwu/data/TIPnovo/systemMHC/all_mzML_files_dic.npy',allow_pickle=True)[()]
    df_filtered = df_filtered.with_columns(
        file=pl.col('Spectrum').str.split(by='.').list.get(0) + '.mzML'
    )
    df_filtered = df_filtered.filter(
        pl.col('file').is_in(all_mzML_files_dic.keys())
    )
    df_filtered = df_filtered.with_columns(
        mzML_path = pl.col('file').replace(all_mzML_files_dic)
    )

if __name__ == '__main__':
    all_mzML_files_dic = np.load('/data/qwu/data/TIPnovo/systemMHC/all_mzML_files_dic.npy', allow_pickle=True)[()]
    mzML_path_list = list(set(all_mzML_files_dic.values()))
    # mzML_path_list = mzML_path_list[:100]
    # mzML_path_list = ["/data/SYSTEMHC/SYSMHC00100/SYSMHC00100_UPN13_class_I_R/work/d9/f2436ebc8ee61fa749815ea6ad9bb9/UPN13_class_I_Rep5.mzML"]
    analyzer_dic = get_precision_parallel(mzML_path_list)
    np.save('/data/qwu/data/TIPnovo/systemMHC/System_tolerance_dic.npy',analyzer_dic)
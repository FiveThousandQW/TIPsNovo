import pandas as pd
import re,os,sys
import random
import polars as pl
import time
from pyteomics import mgf
import numpy as np
from pathlib import Path
import os
import pickle


def extract_spectra_with_index_composite(file_idList):
    """
    使用预先生成的复合索引文件(SCANS, SEQ)，从 MGF 文件中快速提取指定的谱图。
    """
    # --- 修改点：调整参数解析 ---
    # file_idList 现在应包含: [mgf_path, target_pairs_list, output_path]
    mgf_file_path = file_idList[0]
    target_pairs_list = file_idList[1]  # 这是一个元组列表, e.g., [('scan1', 'seq1'), ...]
    output_mgf_path = file_idList[2]

    # 加载复合索引
    index_file_path = mgf_file_path + '.composite.pkl'
    try:
        with open(index_file_path, 'rb') as f:
            index = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Composite index file {index_file_path} not found. Please create it first.")
        return

    # --- 修改点：不再需要 id_to_seq_map ---

    print(f"Extracting {len(target_pairs_list)} spectra from {mgf_file_path} using composite index...")
    found_count = 0

    with open(output_mgf_path, 'w') as mgf_out, open(mgf_file_path, 'r', encoding='utf-8', errors='ignore') as mgf_in:
        # --- 修改点：遍历目标元组列表 ---
        for spec_id, sequence in target_pairs_list:
            composite_key = (spec_id, sequence)

            if composite_key in index:
                mgf_in.seek(index[composite_key])

                # 读取并直接写入整个谱图块，因为它已经包含了SEQ
                for line in mgf_in:
                    mgf_out.write(line)
                    if line.strip() == "END IONS":
                        break

                mgf_out.write("\n")  # 在谱图块后加一个空行
                found_count += 1
            else:
                print(f"Warning: Spectrum pair (SCANS={spec_id}, SEQ={sequence}) not found in the index.")

    print(f"Extraction complete. Found {found_count}/{len(target_pairs_list)} spectra.")

def create_mgf_index_composite(mgf_file_path):
    """
    为 MGF 文件创建一个复合索引，映射 (SCANS, SEQ) 到其在文件中的起始字节位置。
    索引将以 .pkl 格式保存在同一目录下。
    """
    index = {}
    # 为了区分，我们给索引文件加上后缀
    index_file_path = mgf_file_path + '.composite.pkl'

    if os.path.exists(index_file_path):
        print(f"Composite index file {index_file_path} already exists. Skipping.")
        return

    print(f"Creating composite index for {mgf_file_path}...")
    with open(mgf_file_path, 'rb') as f:
        byte_offset = f.tell()
        line = f.readline()
        while line:
            if line.strip() == b'BEGIN IONS':
                current_spectrum_start_offset = byte_offset

                # --- 修改点：初始化两个变量以捕获 SCANS 和 SEQ ---
                spec_id = None
                sequence = None

                # 读取谱图内容以找到ID和序列
                while line.strip() != b'END IONS' and line:
                    # 查找 SCANS
                    if line.startswith(b'SCANS='):
                        try:
                            spec_id = line.strip().split(b'=')[1].decode('utf-8')
                        except (IndexError, UnicodeDecodeError):
                            pass  # 简单忽略解析错误

                    # 查找 SEQ
                    elif line.startswith(b'SEQ='):
                        try:
                            sequence = line.strip().split(b'=')[1].decode('utf-8')
                        except (IndexError, UnicodeDecodeError):
                            pass

                    # 如果两个都找到了，可以提前跳出，提高效率
                    if spec_id and sequence:
                        break

                    line = f.readline()

                # --- 修改点：如果两个都成功找到，则创建复合键 ---
                if spec_id and sequence:
                    composite_key = (spec_id, sequence)
                    index[composite_key] = current_spectrum_start_offset
                else:
                    print(
                        f"Warning: Could not find both SCANS and SEQ for spectrum at offset {current_spectrum_start_offset}.")

            byte_offset = f.tell()
            line = f.readline()

    # 保存索引文件
    with open(index_file_path, 'wb') as f_out:
        pickle.dump(index, f_out)

    print(f"Composite index created and saved to {index_file_path}. Total spectra indexed: {len(index)}")

def SystemMHC():

def ProteomeTools_part1():
    # 收集所有PSM
    root_dir = Path('/data/48/wuqian/Proteometools/part1/result_rename')
    pattern = 'msms.txt'
    files_list = list(root_dir.rglob(pattern))
    lazy_df = pl.scan_csv(
        files_list,
        separator='\t',  # 指定制表符为分隔符，如果你的文件是逗号分隔，则用 ','
        has_header=True,  # 文件包含标题行
        # dtypes={'col_name': pl.Float64} # 如果需要，可以预先指定列类型以提高读取速度
    )
    merged_df = lazy_df.collect()
    merged_df.write_parquet('/data/48/wuqian/Proteometools/part1/part1_allPSM.parquet')

    # 处理修饰格式
    merged_df = pl.read_parquet('/data/48/wuqian/Proteometools/part1/part1_allPSM.parquet')
    merged_df = merged_df.with_columns(
        pl.col('Modified sequence').str.replace_all('M\(ox\)', 'M[147]')
    )
    merged_df = merged_df.with_columns(
        pl.col('Modified sequence').str.replace_all('_', '')
    )
    test = merged_df.filter(pl.col('Modifications')=='4 Oxidation (M)')
    result_nested = merged_df.group_by('Mass analyzer','Fragmentation').count()
    modified_list = merged_df['Modifications'].unique().to_list()

    # 统计analyzer 和 fragmentation
    df_filter  = merged_df.filter(
        (pl.col('Mass analyzer')=="FTMS") & (pl.col('Fragmentation')=="HCD"))
    df_filter.write_parquet('/data/48/wuqian/Proteometools/part1/part1_HCD_FTMS.parquet')

    #
    df_filter = pl.read_parquet('/data/48/wuqian/Proteometools/part1/part1_HCD_FTMS.parquet')


def part3():
    # 处理修饰格式
    merged_df = pl.read_parquet('/data/48/wuqian/Proteometools/part3_HLA/Maxquant_allPSM.parquet')
    merged_df = merged_df.with_columns(
        pl.col('Modified sequence').str.replace_all('M\(ox\)', 'M[147]')
    )
    merged_df = merged_df.with_columns(
        pl.col('Modified sequence').str.replace_all('_', '')
    )
    modified_list = merged_df['Modifications'].unique().to_list()

    # 统计analyzer 和 fragmentation
    # 2042w
    df_filter  = merged_df.filter(
        (pl.col('Mass analyzer')=="FTMS") & (pl.col('Fragmentation')=="HCD"))
    df_filter.write_parquet('/data/48/wuqian/Proteometools/part3_HLA/part2_HCD_FTMS.parquet')

    #
    df_filter = pl.read_parquet('/data/48/wuqian/Proteometools/part1/part1_HCD_FTMS.parquet')


if __name__ == '__main__':
    create_mgf_index_composite('/data/48/wuqian/Proteometools/part1/part1_allPSM.mgf')
    create_mgf_index_composite('/data/48/wuqian/Proteometools/part3_HLA/anno_mgf/part3HLA_best.mgf')
    pass
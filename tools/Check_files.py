import os
import subprocess
import pandas as pd
import re
import sys
from typing import Dict, Any, List, Tuple
import re,os
import multiprocessing
import polars as pl
import random

def check_theMGF():
    file = open('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_FineTurn_train_200wPSM.mgf', 'r').read()
    record_list = file.split('\n\n') #1262033
    PSM_dic = []
    for record in record_list[:-1]:
        record_lines = record.split('\n')
        seq = record_lines[5].split('SEQ=')[1]
        charge = record_lines[2].split('CHARGE=')[1]
        if 'A+' not in charge:
            PSM_dic.append(record)
        else:
            print(record)

    with open('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_FineTurn_train_200wPSM_done.mgf', 'a') as f:
        for sp in PSM_dic:
            f.write(f'{sp}\n\n')

def check_parquetOverlap():
    train_df = pl.read_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_FineTurn_train_200wPSM.parquet')
    test_df = pl.read_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_valid_6w.parquet')
    train_seq_list = train_df['sequence'].to_list()
    test_seq_list = test_df['sequence'].to_list()
    test_df = test_df.filter(
        ~pl.col('sequence').is_in(train_seq_list)
    )
    test_df.write_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_valid_6w.parquet')

    # modified
    train_df = pl.read_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_FineTurn_train_200wPSM.parquet')
    train_df = train_df.filter(
        ~pl.col('sequence').str.contains('\[')
    )
    train_df.write_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_FineTurn_train_200wPSM_unmod.parquet')

def modified_diss():
    df = pl.read_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/part3HLA_best_40wPep.parquet')
    df = df.with_columns(
        pl.col('sequence').str.replace('M[UNIMOD:35]','M[147]', literal=True)
    )
    df.write_parquet('/data/48/wuqian/fast/TipsNovo/data/readyData/part3HLA_best_40wPep.parquet')


def the_percursorPeptide_num():
    parquet_path = '/data/qwu/data/TIPnovo/data/Proteometools_train.parquet'
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        (pl.col('sequence') + '_' + pl.col('precursor_charge').cast(pl.Utf8) ).alias('seq_charge')
    )
    print(f"Peptide number: {len(df['sequence'].unique())}")
    print(f"precursor number: {len(df['seq_charge'].unique())}")


    # /data/qwu/data/TIPnovo/data/SystemMHC_valid_6w.parquet
    # Peptide number: 60703
    # precursor number: 66092

    # /data/qwu/data/TIPnovo/data/SystemMHC_train_40w.parquet
    # Peptide number: 412674
    # precursor number: 484763

    # /data/qwu/data/TIPnovo/data/Proteometools_train.parquet
    # Peptide number: 350000
    # precursor number: 421956


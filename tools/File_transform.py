import subprocess
import numpy as np
import pandas as pd
import re,os
import multiprocessing
import torch
import polars as pl

def raw2mgf_mzML():
    def RawTomgf_mzML(file_path,output_path):
        order = f'mono /home/qwu/soft/ThermoRawFileParse/ThermoRawFileParser.exe -i {file_path} -o {output_path} --format=0'
        print(order)
        subprocess.run(order, shell=True)

        # subprocess.run(f'mkdir {sample_path}/mzML', shell=True)
        # order = f'mono /home/qwu/soft/ThermoRawFileParse/ThermoRawFileParser.exe -i {sample_path}/raw/{file} -o {sample_path}/mzML --format=1'
        # subprocess.run(order, shell=True)

    sample_list = os.listdir(f'/data/48/wuqian/Proteometools/part2/raw')
    pool = multiprocessing.Pool(40)
    for i in sample_list:
        sample_path = f'/data/48/wuqian/Proteometools/part2/raw/{i}'
        output_path = f'/data/48/wuqian/Proteometools/part2'
        pool.apply_async(func=RawTomgf_mzML, args=(sample_path, output_path,))
    pool.close()
    pool.join()

def mgf_test(file_path):
    train_df = pl.read_parquet("/data/48/wuqian/fast/TipsNovo/data/systemMHC_train112W_anno.parquet")
    valid_df = pl.read_parquet("/data/48/wuqian/fast/TipsNovo/data/systemMHC_valid7w.parquet")

    train_seq_list = train_df['sequence'].unique().to_list()
    valid_seq_list = valid_df['sequence'].unique().to_list()
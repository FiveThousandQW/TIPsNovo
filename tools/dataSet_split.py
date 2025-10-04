import pandas as pd
import re,os,sys
import random
import polars as pl
import time
from pyteomics import mgf
import numpy as np


def extract_spectra_from_mgf(input_file, output_file, target_scans):
    """
    优化版的MGF谱图提取函数。
    - 使用I/O缓冲批量写入以提高速度。
    """
    print(f"开始处理文件: {input_file} (优化版)")
    print(f"共需要查找 {len(target_scans)} 个目标谱图。")

    found_count = 0
    # 写入缓冲区大小
    WRITE_BUFFER_SIZE = 1000

    # 预定义常量
    BEGIN_IONS = 'BEGIN IONS'
    END_IONS = 'END IONS'
    SCANS_PREFIX = 'SCANS='

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
                open(output_file, 'w', encoding='utf-8') as fout:

            current_spectrum = []
            write_buffer = []
            is_in_spectrum_block = False
            found_target_in_block = False

            for line in fin:
                # 仅在需要时执行strip()
                if line[0] == 'B' and line.strip() == BEGIN_IONS:
                    is_in_spectrum_block = True
                    found_target_in_block = False
                    current_spectrum = [line]
                    continue

                if is_in_spectrum_block:
                    current_spectrum.append(line)

                    # 优先检查最可能出现的行，比如 END IONS
                    if line[0] == 'E' and line.strip() == END_IONS:
                        if found_target_in_block:
                            write_buffer.extend(current_spectrum)
                            found_count += 1

                            if len(write_buffer) >= WRITE_BUFFER_SIZE * 10:  # 估算行数
                                fout.writelines(write_buffer)
                                write_buffer = []

                        is_in_spectrum_block = False
                        current_spectrum = []

                    # 检查SCANS行
                    elif not found_target_in_block and line.startswith(SCANS_PREFIX):
                        scan_id = line[6:].strip()  # 直接切片比split更快
                        if scan_id in target_scans:
                            found_target_in_block = True

            # 清空剩余的缓冲区
            if write_buffer:
                fout.writelines(write_buffer)

    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_file}' 未找到。")
        return
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        return

    print("\n----------------------------------------")
    print("处理完成！")
    print(f"总共在文件中找到了 {found_count} 个匹配的谱图。")
    print(f"结果已保存到文件: {output_file}")
    print("----------------------------------------")

def extract_with_pyteomics(input_file, output_file, target_scans):
    """
    使用 pyteomics 库高效提取MGF谱图 (已修正导入问题)。
    """
    print(f"开始使用 pyteomics 处理文件: {input_file}")
    start_time = time.time()

    def generate_spectra():
        # 2. 调用方式从 pyteomics.mgf.MGF 改为 mgf.MGF
        with mgf.MGF(input_file) as reader:
            for spectrum in reader:
                scan_id = spectrum['params'].get('scans')
                # 确保scan_id存在且不为空
                if scan_id:
                    # pyteomics 0.5.1 版本后，scans可能是一个字符串而不是列表
                    # 为了兼容性，我们统一转为字符串处理
                    scan_str = str(scan_id[0] if isinstance(scan_id, list) else scan_id)
                    if scan_str in target_scans:
                        yield spectrum

    # 3. 调用方式从 pyteomics.mgf.write 改为 mgf.write
    found_count = mgf.write(generate_spectra(), output=output_file)

    end_time = time.time()
    print("\n----------------------------------------")
    print("处理完成！")
    print(f"总共在文件中找到了 {found_count} 个匹配的谱图。")
    print(f"结果已保存到文件: {output_file}")
    print(f"任务耗时: {end_time - start_time:.2f} 秒")
    print("----------------------------------------")

def split_data():
    # 将126w张谱图进行分割，使得validation与train dataset在peptide层面上没有overlap
    file = open('/data/48/wuqian/Proteometools/part1/anno_mgf/1461raw_high.mgf', 'r').read()
    record_list = file.split('\n\n') #1262033
    peptides_dic = {}
    for record in record_list[:-1]:
        record_lines = record.split('\n')
        seq = record_lines[5].split('SEQ=')[1]
        charge = record_lines[1].split('CHARGE=')[1]
        if seq in peptides_dic.keys():
            peptides_dic[f'{seq}'].append(record)
        else:
            peptides_dic[f'{seq}'] = [record]

    peptide_key_list = list(peptides_dic.keys())
    random.shuffle(peptide_key_list)
    trainset = peptide_key_list[:350000]
    validset = peptide_key_list[350000:]
    with open('/data/48/wuqian/Proteometools/part1/anno_mgf/1461_trainValid_split/shuffle/train.mgf', 'a') as f:
        for seq in trainset:
            peptide_sp_list = peptides_dic[f'{seq}']
            for sp in peptide_sp_list:
                f.write(f'{sp}\n\n')

    with open('/data/48/wuqian/Proteometools/part1/anno_mgf/1461_trainValid_split/shuffle/valid.mgf','a') as f:
        for seq in validset:
            peptide_sp_list = peptides_dic[f'{seq}']
            for sp in peptide_sp_list:
                f.write(f'{sp}\n\n')

def merge_msms():
    from pathlib import Path
    root_dir = Path('/data/48/wuqian/Proteometools/part2/unzip_result')

    # 2. 指定命名规则：所有以 .csv 结尾的文件
    pattern = 'msms.txt'

    # 3. 使用 rglob() 递归查找
    print(f"在 '{root_dir}' 及其子文件夹中查找 '{pattern}':")
    files_list = list(root_dir.rglob(pattern))
    lazy_df = pl.scan_csv(
        files_list,
        separator='\t',  # 指定制表符为分隔符，如果你的文件是逗号分隔，则用 ','
        has_header=True,  # 文件包含标题行
        # dtypes={'col_name': pl.Float64} # 如果需要，可以预先指定列类型以提高读取速度
    )
    merged_df = lazy_df.collect()
    merged_df.write_parquet('/data/48/wuqian/Proteometools/part2/Maxquant_allPSM.parquet')

    merged_df = merged_df.filter(
        (pl.col('Mass analyzer')=="FTMS")& (pl.col('Fragmentation')=="HCD")
    )
    merged_df = merged_df.with_columns(
        (pl.col('Raw file') + '.mgf').alias('MGF')
    )
    mgf_list = os.listdir('/data/48/wuqian/Proteometools/part2/mgf')
    merged_df = merged_df.filter(
        merged_df['MGF'].is_in(mgf_list)
    )
    part13_df = pl.read_parquet('/data/48/wuqian/Proteometools/allPart/part13_HCD_FTMS_PSM.parquet')
    part123_df = pl.concat([part13_df, merged_df])
    part123_df.write_parquet('/data/48/wuqian/Proteometools/allPart/partALL_HCD_FTMS_PSM.parquet')

def select_mgf():
    MSMSmerged_df = pl.read_parquet('/data/48/wuqian/Proteometools/part3_HLA/Maxquant_allPSM.parquet')
    MSMSmerged_df = MSMSmerged_df.filter(
        pl.col('Mass analyzer') == "FTMS"
    )
    MSMSmerged_df = MSMSmerged_df.with_columns(
        (pl.col('Raw file') + ':' + pl.col('Scan number').cast(pl.Utf8)).alias('file_scan_id')
    )
    peptide_list = MSMSmerged_df['Sequence'].unique().to_list()
    sample_peptide_list = random.sample(peptide_list, 400000)
    df_filter = MSMSmerged_df.filter(
        pl.col('Sequence').is_in(sample_peptide_list)
    )
    df_filter= df_filter.group_by('Sequence').agg(
        pl.all().head(4)
    )
    columns_to_explode = [col for col in df_filter.columns if df_filter.schema[col] == pl.List]
    df_filter = df_filter.explode(columns_to_explode)

    # 99w -> 44w
    # df_filter.write_csv('/data/48/wuqian/fast/TipsNovo/data/part3HLA_best_40wPepNum.csv')
    scan_id_set = set(df_filter['file_scan_id'].unique().to_list())
    extract_spectra_from_mgf(
        input_file='/data/48/wuqian/fast/TipsNovo/data/part3HLA_best.mgf',
        output_file='/data/48/wuqian/fast/TipsNovo/data/part3HLA_best_40wPep.mgf',
        target_scans=scan_id_set
    )

def SystemMHC_tolerance():
    System_analyzer_dic = np.load('/data/qwu/data/TIPnovo/systemMHC/System_analyzer_dic.npy',allow_pickle=True)[()]
    System_analyzer_df = {'path':list(System_analyzer_dic.keys()),'analyzer':list(System_analyzer_dic.values())}
    System_analyzer_df = pl.DataFrame(System_analyzer_df)

if __name__ == '__main__':
    select_mgf()
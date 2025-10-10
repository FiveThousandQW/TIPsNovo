import polars as pl
import numpy as np
import re,os,glob
import multiprocessing
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools.spectrum2parquet import convert_mgf_to_parquet
import itertools



def process_single_mgf(mgf_path, scans_to_annotate):
    """
    (优化版) 处理单个MGF文件，提取并注释指定的SCANS。

    Args:
        mgf_path (str): MGF文件的路径。
        scans_to_annotate (dict): 一个字典 {scan_id: sequence}。

    Returns:
        str: 包含所有从该文件中提取并注释过的谱图的字符串。
             如果文件不存在或未找到任何匹配的谱图，则返回空字符串。
    """
    if not os.path.exists(mgf_path):
        # 警告信息在主进程中打印更清晰，这里只返回空
        return ""

    # OPTIMIZATION 1: 一次性将键转换为字符串集合，用于O(1)复杂度的快速查找。
    # 我们假设从文件中读取的 SCANS 总是字符串，因此保持类型一致。
    # 集合（set）的查找速度和字典一样快。
    target_scans_set = set(str(k) for k in scans_to_annotate.keys())

    # OPTIMIZATION 2: 确保查找字典的键也是字符串。
    scans_to_annotate_str_keys = {str(k): v for k, v in scans_to_annotate.items()}

    annotated_spectra = []
    try:
        with open(mgf_path, 'r', encoding='utf-8') as f:
            current_spectrum_lines = []
            current_scan = None
            in_spectrum_block = False

            for line in f:
                stripped_line = line.strip()

                if stripped_line == "BEGIN IONS":
                    in_spectrum_block = True
                    current_spectrum_lines = [line]
                    current_scan = None
                    continue  # 直接开始下一次循环

                if not in_spectrum_block:
                    continue

                # --- 以下代码只在 in_spectrum_block 为 True 时执行 ---

                if stripped_line.startswith("TITLE="):
                    # 保留您的自定义TITLE逻辑
                    file_name_without_ext = os.path.splitext(os.path.basename(mgf_path))[0]
                    current_spectrum_lines.append(f"TITLE={file_name_without_ext}\n")

                elif stripped_line.startswith("SCANS="):
                    current_scan = stripped_line.split('=')[-1].strip()
                    current_spectrum_lines.append(line)

                elif stripped_line == "END IONS":
                    current_spectrum_lines.append(line)
                    # OPTIMIZATION 3: 直接在集合中查找，速度极快！
                    if current_scan and current_scan in target_scans_set:
                        sequence = scans_to_annotate_str_keys[current_scan]
                        seq_line = f"SEQ={sequence}\n"

                        # OPTIMIZATION 4 (Code Quality): 插入到倒数第二行更稳健
                        # 您的 insert(3, ...) 写法可能因MGF格式变化而出错
                        current_spectrum_lines.insert(3, seq_line)
                        annotated_spectra.append("".join(current_spectrum_lines))

                    # 重置状态
                    in_spectrum_block = False

                else:
                    # 将其他所有行（如PEPMASS, CHARGE, 峰列表等）加入
                    current_spectrum_lines.append(line)

    except Exception as e:
        print(f"处理文件 {mgf_path} 时发生错误: {e}")
        return ""

    return "".join(annotated_spectra)

def extract_and_annotate_spectra_mp(mgf_data_dict, output_mgf_path, num_processes=None):
    """
    使用多进程从多个MGF文件中提取并注释指定的谱图。

    Args:
        mgf_data_dict (dict): 包含MGF文件路径和对应SCANS信息的字典。
            格式: {"path/to/file.mgf": {"scan_id": "SEQUENCE", ...}, ...}
        output_mgf_path (str): 输出结果的MGF文件路径。
        num_processes (int, optional): 使用的进程数。如果为None，则默认为CPU核心数。
    """
    if num_processes is None:
        # os.cpu_count() 在某些环境下可能返回None，提供一个备选项
        num_processes = os.cpu_count() or 4

    print(f"使用 {num_processes} 个进程开始处理 {len(mgf_data_dict)} 个MGF文件...")

    # 创建任务列表
    tasks = [(mgf_path, scans) for mgf_path, scans in mgf_data_dict.items()]

    all_results = []

    # 使用 concurrent.futures.ProcessPoolExecutor，它更现代且易于使用
    # tqdm 用于显示进度条
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # submit任务
        future_to_task = {executor.submit(process_single_mgf, *task): task for task in tasks}

        # as_completed在任务完成时获取结果，可以更好地更新进度条
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="处理MGF文件"):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                task_info = future_to_task[future]
                print(f"处理文件 {task_info[0]} 时子进程抛出异常: {e}")
    print("所有文件处理完毕，正在将结果写入输出文件...")
    try:
        with open(output_mgf_path, 'w', encoding='utf-8') as f_out:
            # 批量写入以提高效率
            f_out.writelines(all_results)
        print(f"任务完成！结果已保存至: {output_mgf_path}")
    except IOError as e:
        print(f"写入输出文件 {output_mgf_path} 时失败: {e}")

def getMGF(input_parquet,output_file,output_parquet,
           num_processes=32,
           parent_path='/data/48/wuqian/Proteometools/allPart/mgf'):
    filter_df = pl.read_parquet(input_parquet)
    # 输入一个字典，如{"02445d_BA12-TUM_HLA_12_01_01-3xHCD-1h-R4.mgf":{'1232':'AASCD','516':'ADWF'}}

    MGF_SCANanno_dic = {}
    for row in filter_df.iter_rows(named=True):
        mgf_path = os.path.join(parent_path, row['MGF'])
        scan_id = row['Scan number']
        seq = row['Modified sequence']
        if mgf_path not in MGF_SCANanno_dic.keys():
            MGF_SCANanno_dic[mgf_path] = {scan_id: seq}
        else:
            MGF_SCANanno_dic[mgf_path][scan_id] = seq

    # 4. 调用主函数运行
    extract_and_annotate_spectra_mp(MGF_SCANanno_dic, output_file, num_processes=num_processes)
    # extract_and_annotate_spectra_mp(test_dic, output_file, num_processes=num_processes)
    print('将mgf 转换成parquet文件')
    convert_mgf_to_parquet(output_file,output_parquet)

def optimized_trainingValid_check(
    train_parquet: str,
    output_train_parquet: str,
    valid_parquet: str = '/data/48/wuqian/fast/TipsNovo/data/readyData/SystemMHC_valid_6w.parquet'
):

    # 1. 使用 scan_parquet 启动惰性计算，几乎不消耗内存
    train_lazy_df = pl.scan_parquet(train_parquet)
    valid_lazy_df = pl.scan_parquet(valid_parquet)

    # 2. 从验证集中只选择我们关心的'sequence'列，并去重以优化join性能
    valid_sequences = valid_lazy_df.select('sequence').unique()

    # 3. 执行 anti-join 操作
    # 这会保留 train_lazy_df 中'sequence'不在 valid_sequences 中的所有行
    filtered_lazy_df = train_lazy_df.join(
        valid_sequences,
        on='sequence',
        how='anti'
    )
    filtered_lazy_df.collect().write_parquet(output_train_parquet)

def fast_preprocess():
    df = pl.scan_parquet('/data/48/wuqian/Proteometools/allPart/partALL_HCD_FTMS_PSM.parquet')
    df = df.with_columns(
        pl.col('Modified sequence').str.replace_all('M\(ox\)', 'M[147]')
    )
    df = df.with_columns(
        pl.col('Modified sequence').str.replace_all('_', '')
    )
    df = df.with_columns(
        (pl.col('Modified sequence') + ':' + pl.col('Charge').cast(pl.Utf8)).alias('Precursor')
    )
    # peptide_count = df['Sequence'].unique().to_list()
    # precursor_count = df['Precursor'].unique().to_list()
    # columns_to_explode = [col for col in df.columns if col != "Precursor"]
    best_practice_query = (
        df.lazy()
        .sort("Score", descending=True)  # 1. 首先全局排序
        .group_by("Precursor", maintain_order=True)  # 2. 然后分组，maintain_order=True 确保组内顺序不变
        .head(2)  # 3. 取出每个组的头两条记录
        .with_columns(
            pl.col('Precursor').str.replace_all('M(ox)', 'M[147]', literal=True)
        )
    )
    best_practice_query = best_practice_query.collect()
    best_practice_query.write_parquet(
        '/data/48/wuqian/Proteometools/allPart/partALL_allPep_312wPSM.parquet'
    )



if __name__ == '__main__':
    input_parquet = '/data/48/wuqian/Proteometools/allPart/partALL_allPep_312wPSM.parquet'
    output_file = "/data/48/wuqian/Proteometools/allPart/spectrum/part13_allPep_262wPSM.mgf"
    output_parquet = "/data/48/wuqian/Proteometools/allPart/spectrum/part13.parquet"
    getMGF(input_parquet,output_file,output_parquet)
    clear_training = "/data/48/wuqian/Proteometools/allPart/spectrum/part13_clear.parquet"
    print('去除与valid的重复')
    optimized_trainingValid_check(output_parquet,clear_training)
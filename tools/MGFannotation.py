import os
import subprocess
import pandas as pd
import re
import sys
from typing import Dict, Any, List, Tuple, overload
import re,os
import multiprocessing
import polars as pl

def format_sequence(mod_sequence: str) -> str:
    """
    格式化MaxQuant的修饰序列。
    例如，将 "_PEPT(mod)IDE_" 转换为 "PEPT(mod)IDE"。
    """
    if mod_sequence.startswith('_') and mod_sequence.endswith('_'):
        return mod_sequence[1:-1]
    return mod_sequence


def parse_msms_file(msms_path: str,First_PSM) -> Dict[int, Dict[str, Any]]:
    """
    解析 msms.txt 文件并返回一个以扫描号为键的字典。

    Args:
        msms_path: msms.txt 文件的路径。

    Returns:
        一个字典，将扫描号映射到包含序列、电荷和原始文件名的信息。
    """
    try:
        # 增加对可能存在的解析错误的鲁棒性
        msms_df = pd.read_csv(msms_path, sep='\t', on_bad_lines='warn')
    except Exception as e:
        print(f"错误：无法读取或解析 msms.txt 文件: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = ['Scan number', 'Modified sequence', 'Charge', 'Raw file']
    if not all(col in msms_df.columns for col in required_cols):
        print(f"错误：msms.txt 文件缺少必要的列。需要: {required_cols}", file=sys.stderr)
        sys.exit(1)

    if First_PSM:
        msms_df = msms_df.sort_values(by=['Score'], ascending=False)
        msms_df = msms_df.drop_duplicates(subset=['Modified sequence'])

    identifications = {}
    for _, row in msms_df.iterrows():
        try:
            scan_num = int(row['Scan number'])
            identifications[scan_num] = {
                'seq': row['Modified sequence'][1:-1].replace('M(ox)','M[UNIMOD:35]'),
                'charge': row['Charge'],
                'raw_file': row['Raw file']
            }
        except (ValueError, TypeError):
            # 忽略无法正确解析扫描号的行
            continue

    return identifications


def process_mgf_file(mgf_path: str, output_path: str, identifications: Dict[int, Dict[str, Any]]):
    """
    处理MGF文件，根据鉴定结果进行注释，并写入新的MGF文件。

    Args:
        mgf_path: 输入的MGF文件路径。
        output_path: 输出的注释后MGF文件路径。
        identifications: 从msms.txt解析出的鉴定信息字典。
    """
    print(f"开始处理 MGF 文件: {mgf_path}...")
    annotated_spectra_count = 0

    with open(mgf_path, 'r') as f_in, open(output_path, 'w') as f_out:
        current_spectrum_lines: List[str] = []
        in_spectrum = False

        for line in f_in:
            stripped_line = line.strip()

            if stripped_line == 'BEGIN IONS':
                in_spectrum = True
                current_spectrum_lines = [line]
            elif stripped_line == 'END IONS':
                if not in_spectrum:
                    continue

                in_spectrum = False
                current_spectrum_lines.append(line)

                # 提取信息并处理谱图
                scan_num = -1
                header_lines: List[str] = []
                peak_data: List[Tuple[float, float]] = []

                for spec_line in current_spectrum_lines:
                    # 查找扫描号 (scan= or SCANS=)
                    match = re.match(r"SCANS?=(\d+)", spec_line.strip(), re.IGNORECASE)
                    if match:
                        scan_num = int(match.group(1))

                    # 分离头信息和峰数据
                    if re.match(r'^\d+.*\s+\d+', spec_line.strip()):
                        parts = spec_line.strip().split()
                        if len(parts) >= 2:
                            try:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                peak_data.append((mz, intensity))
                            except ValueError:
                                continue  # 忽略格式不正确的峰行
                    elif not (spec_line.strip() == 'BEGIN IONS' or spec_line.strip() == 'END IONS'):
                        header_lines.append(spec_line)

                # 如果扫描号在鉴定结果中，则进行注释并写入文件
                if scan_num != -1 and scan_num in identifications:
                    annotated_spectra_count += 1
                    id_data = identifications[scan_num]

                    # 提取原始的PEPMASS和RTINSECONDS
                    pepmass_line = next((l for l in header_lines if l.upper().startswith("PEPMASS=")), "")
                    rt_line = next((l for l in header_lines if l.upper().startswith("RTINSECONDS=")), "")

                    # 写入新的注释头信息
                    f_out.write('BEGIN IONS\n')
                    f_out.write(f"CHARGE={id_data['charge']}+\n")
                    f_out.write(f"SCANS={id_data['raw_file']}:{scan_num}\n")
                    if rt_line:
                        f_out.write(rt_line)
                    if pepmass_line:
                        f_out.write(pepmass_line)

                    formatted_seq = format_sequence(id_data['seq'])
                    f_out.write(f"SEQ={formatted_seq}\n")

                    # 归一化处理并写入峰数据
                    # for mz, intensity in peak_data:
                    #     normalized_intensity = intensity
                    #     f_out.write(f"{mz} {normalized_intensity}\n")

                    if peak_data:  # 确保峰列表不为空
                        # 1. 找到基峰的强度（即最大强度）
                        max_intensity = max(p[1] for p in peak_data)

                        if max_intensity > 0:
                            # 2. 用每个峰的强度除以最大强度
                            for mz, intensity in peak_data:
                                normalized_intensity = intensity / max_intensity
                                f_out.write(f"{mz} {normalized_intensity}\n")
                    f_out.write('END IONS\n\n')  # 在谱图之间留一个空行

            elif in_spectrum:
                current_spectrum_lines.append(line)

    print(f"处理完成。共注释了 {annotated_spectra_count} 张谱图。")
    print(f"输出文件已保存至: {output_path}")


def mgf_annotation_main(msms_file_path, mgf_file_path, output_mgf_path,First_PSM=True):
    # msms_file_path = '/data/48/wuqian/Proteometools/test/Result/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/msms.txt'
    # mgf_file_path = '/data/48/wuqian/Proteometools/test/mgf/01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mgf'  # 您的未注释的MGF文件名
    # output_mgf_path = '/data/48/wuqian/Proteometools/test/mgf/annotated_output.mgf'

    # 执行主逻辑
    identifications_dict = parse_msms_file(msms_file_path,First_PSM)
    if identifications_dict:
        process_mgf_file(mgf_file_path, output_mgf_path, identifications_dict)

def Proteometools_part1():
    Result_path2 = '/data/48/wuqian/Proteometools/part2/unzip_result'
    Result_path = '/data/48/wuqian/Proteometools/part2/result'
    mgf_path = '/data/48/wuqian/Proteometools/part2/mgf'
    result_list = [x for x in os.listdir(Result_path) if '.zip' in x]
    mgf_list = [x for x in os.listdir(mgf_path) if '.mgf' in x]

    # 将mgf对应到Result
    result_mgf_dic = {}
    for mgf in mgf_list:
        result_name = mgf.replace('.mgf','.zip')
        # result_name = result_name.replace('-', '_', 1)
        result_mgf_dic[mgf] = result_name
    for mgf,i in result_mgf_dic.items():
        if i not in result_list:
            result_mgf_dic[mgf] = i.replace('-tryptic',r'-unspecific')

    for mgf,i in result_mgf_dic.items():
        if i not in result_list:
            print(mgf,i)

    # 转移文件，将Result改名
    for mgf_file,result_file in result_mgf_dic.items():
        print(mgf_file,result_file)
        subprocess.run(f'cp {Result_path}/{result_file} {Result_path2}/{mgf_file.replace(".mgf",".zip")}', shell=True)

    # 从结果中解压出对应的结果文件
    for mgf in mgf_list:
        subprocess.run(f'mkdir {Result_path2}/{mgf.replace(".mgf","")}', shell=True)
        subprocess.run(f'unzip {Result_path2}/{mgf.replace(".mgf",".zip")} -d {Result_path2}/{mgf.replace(".mgf","")}', shell=True)

    pool = multiprocessing.Pool(80)
    for i in mgf_list:
        msms_file_path = f'/data/48/wuqian/Proteometools/test/Result/{i.replace(".mgf","")}/msms.txt'
        mgf_file_path = f'/data/48/wuqian/Proteometools/test/mgf/{i}'  # 您的未注释的MGF文件名
        output_mgf_path = f'/data/48/wuqian/Proteometools/test/anno_mgf/all/{i}'
        First_PSM = False
        pool.apply_async(func=mgf_annotation_main, args=(msms_file_path, mgf_file_path, output_mgf_path,First_PSM,))
    pool.close()
    pool.join()

def Proteometools_part3HLA():
    Result_path2 = '/data/48/wuqian/Proteometools/part3_HLA/result_unzip'
    Result_path = '/data/48/wuqian/Proteometools/part3_HLA/result'
    mgf_path = '/data/48/wuqian/Proteometools/part3_HLA/mgf'
    result_list = [x for x in os.listdir(Result_path) if '.zip' in x]
    mgf_list = [x for x in os.listdir(mgf_path) if '.mgf' in x]

    # 将mgf对应到Result
    result_mgf_dic = {}
    for mgf in mgf_list:
        result_name = "T"+ mgf.split('-T')[1].replace('.mgf','-unspecific.zip')
        result_name = result_name.replace('-', '_', 1)
        result_mgf_dic[mgf] = result_name

    nofind_list = []
    for mgf,i in result_mgf_dic.items():
        if i not in result_list:
            tmp_name1 = i.replace('-unspecific',r'-semitryptic')
            tmp_name2 = i.replace('-unspecific', r'-AspN')
            tmp_name3 = i.replace('-unspecific', r'-LysN')
            tmp_name_set = set([tmp_name1, tmp_name2, tmp_name3])
            overlap = tmp_name_set.intersection(set(result_list))
            if overlap:
                result_mgf_dic[mgf] = list(overlap)[0]
            else:
                nofind_list.append(mgf)

    for mgf,i in result_mgf_dic.items():
        if i not in result_list:
            print(mgf,i)

    # 转移文件，将Result改名
    for mgf_file,result_file in result_mgf_dic.items():
        print(mgf_file,result_file)
        subprocess.run(f'ln {Result_path}/{result_file} {Result_path2}/{mgf_file.replace(".mgf",".zip")}', shell=True)

    # 从结果中解压出对应的结果文件
    for mgf in mgf_list:
        subprocess.run(f'mkdir {Result_path2}/{mgf.replace(".mgf","")}', shell=True)
        subprocess.run(f'unzip {Result_path2}/{mgf.replace(".mgf",".zip")} -d {Result_path2}/{mgf.replace(".mgf","")}', shell=True)

    pool = multiprocessing.Pool(80)
    for i in mgf_list:
        msms_file_path = f'/data/48/wuqian/Proteometools/part3_HLA/result_unzip/{i.replace(".mgf","")}/msms.txt'
        mgf_file_path = f'/data/48/wuqian/Proteometools/part3_HLA/mgf/{i}'  # 您的未注释的MGF文件名
        output_mgf_path = f'/data/48/wuqian/Proteometools/part3_HLA/anno_mgf/all/{i}'
        First_PSM = False
        pool.apply_async(func=mgf_annotation_main, args=(msms_file_path, mgf_file_path, output_mgf_path,First_PSM,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    pass
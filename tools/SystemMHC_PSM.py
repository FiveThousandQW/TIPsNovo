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

FTMS_KEYWORDS = ['orbitrap', 'fourier transform', 'ftms']
ITMS_KEYWORDS = ['ion trap', 'itms']
TOF_KEYWORDS = ['time-of-flight', 'tof']

def get_single_mzml_analyzer_type(mzml_path: str) -> tuple[str, str]:
    """
    工作函数：处理单个mzML文件，返回其分析器类型。
    设计为在独立的进程中运行。

    Args:
        mzml_path (str): 单个mzML文件的路径。

    Returns:
        tuple[str, str]: 一个元组，包含(文件路径, 分析器类型)。
                         分析器类型为 "FTMS", "ITMS", 或 "Unknown"。
    """
    try:
        # 使用 with 语句确保文件正确关闭
        # iterparse=True 开启迭代解析模式，非常节省内存
        with mzml.MzML(mzml_path, iterparse=True) as reader:
            # 我们只需要检查第一个谱图的元数据就足够了，这极大地提高了速度
            first_spectrum = next(reader, None)

            if first_spectrum is None:
                return (mzml_path, "Empty_or_Error")

            # 将谱图的所有元信息合并成一个长字符串，方便搜索
            # .keys() 和 .values() 包含了所有CV a nd user params
            metadata_text = str(first_spectrum.keys()).lower() + str(first_spectrum.values()).lower()
            # print(metadata_text)
            # 检查关键词
            if any(keyword in metadata_text for keyword in FTMS_KEYWORDS):
                return (mzml_path, "FTMS")
            elif any(keyword in metadata_text for keyword in ITMS_KEYWORDS):
                return (mzml_path, "ITMS")
            elif any(keyword in metadata_text for keyword in TOF_KEYWORDS):
                return (mzml_path, "TOF")
            else:
                return (mzml_path, "Unknown")

    except FileNotFoundError:
        return (mzml_path, "File_Not_Found")
    except Exception:
        # 捕获其他可能的解析错误
        return (mzml_path, "Parsing_Error")


def get_analyzer_types_parallel(mzml_paths: list[str]) -> dict[str, str]:
    """
    主函数：使用并行处理高效地获取大量mzML文件的分析器类型。

    Args:
        mzml_paths (list[str]): 包含数千万个mzML文件路径的列表。

    Returns:
        dict[str, str]: 一个字典，键是mzML文件路径，值是其分析器类型。
    """
    # os.cpu_count() 获取机器的CPU核心数，创建对应数量的进程
    # 也可以手动指定 max_workers=16 等
    num_workers = 16
    print(f"启动 {num_workers} 个工作进程来处理 {len(mzml_paths)} 个文件...")

    results_dict = {}

    # ProcessPoolExecutor 是实现多进程的推荐方式
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # executor.map 的方式更简洁，但tqdm的total参数可能不精确
        # 我们使用 submit 和 as_completed，这样可以更精确地控制进度条
        futures = {executor.submit(get_single_mzml_analyzer_type, path): path for path in mzml_paths}

        # 使用tqdm创建进度条
        for future in tqdm(as_completed(futures), total=len(mzml_paths), desc="Processing mzML files"):
            try:
                path, analyzer_type = future.result()
                results_dict[path] = analyzer_type
            except Exception as e:
                # 如果工作函数本身出现无法捕获的严重错误
                path = futures[future]
                results_dict[path] = f"Critical_Error: {e}"

    return results_dict


def getSystem_dir(path):
    dir_list = os.listdir(path)
    SystemMHC_items_list = []
    for item in dir_list:
        if 'SYSMHC00' in item:
            SystemMHC_items_list.append(item)
    return SystemMHC_items_list

def count_lines_iter(filepath):
    """
    通过迭代文件对象来计算行数，内存效率最高。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                count += 1
        return count
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
        return -1
    except Exception as e:
        print(f"发生错误: {e}")
        return -1

def getSample_inDataset(path):
    All_PSM_path = '/data/qwu/data/TIPnovo/systemMHC'
    sample_list = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
    sample_data_list = []
    dataset_size = 0
    for sample in sample_list:
        sample_path = os.path.join(path, sample)
        result_path = os.path.join(sample_path, 'Results','Peplevel_FDR','psm_result.csv')
        if Path(result_path).is_file() and Path(result_path).stat().st_size >1000:
            sample_data_list.append(sample)
            dataset_size += count_lines_iter(result_path)
            subprocess.run(f'cp {result_path} {os.path.join(All_PSM_path,os.path.basename(path)+ "___" + sample)}.PSM', shell=True)
    if len(sample_data_list) > 0:
        return sample_data_list,dataset_size
    else:
        return None,None

def getHLAbinderPeptide():
    path = '/data/SYSTEMHC'
    SystemMHC_items_list = getSystem_dir(path)
    Allele_peptide_dic = {'Allele':[],'Peptide':[]}
    for dataset in SystemMHC_items_list:
        print(dataset)
        try:
            sample_data_list, dataset_size = getSample_inDataset(os.path.join(path, dataset))
            if not sample_data_list:
                continue
            for sample in sample_data_list:
                HLA_path = os.path.join('/data/SYSTEMHC',dataset,sample,'Results/MHCmotifs/HLAspecificmotifs')
                if not os.path.isdir(HLA_path):
                    continue
                allele_dir_list = os.listdir(HLA_path)
                for allele in allele_dir_list:
                    single_allele_list = [x for x in os.listdir(os.path.join(HLA_path,allele)) if '.txt' in x]
                    for single_allele in single_allele_list:
                        single_allele_path = os.path.join(HLA_path,allele,single_allele)
                        peptide_list = [x.strip() for x in open(single_allele_path,'r').readlines()]
                        Allele_peptide_dic['Peptide'] = Allele_peptide_dic['Peptide'] + peptide_list
                        Allele_peptide_dic['Allele'] = Allele_peptide_dic['Allele'] + [allele]*len(peptide_list)
        except:
            print(f'Error in {dataset}')

    base_path = '/data/SYSTEMHC'
    path_obj = Path(base_path)
    alleles = []
    peptides = []

    # 使用 rglob ('**/') 来递归查找所有匹配的 .txt 文件
    # 模式：/data/SYSTEMHC/*/Results/MHCmotifs/HLAspecificmotifs/*/*.txt
    #       <base_path> /<dataset>/<...>   /<allele_dir>  /<file>
    file_pattern = "*/Results/MHCmotifs/HLAspecificmotifs/*/*.txt"

    print("Finding all matching peptide files...")
    # rglob 会返回一个生成器，内存效率高
    all_peptide_files = list(path_obj.rglob(file_pattern))
    print(f"Found {len(all_peptide_files)} files to process.")

    for idx,file_path in enumerate(all_peptide_files):
        print(idx)
        try:
            # 从路径中直接提取 allele 名称
            # file_path.parts 会返回路径的各个部分，例如：
            # ('data', 'SYSTEMHC', 'dataset_A', ..., 'HLAspecificmotifs', 'HLA-A*01:01', 'peptides.txt')
            # allele 名称是倒数第二个部分
            allele = file_path.parts[-2]

            with open(file_path, 'r') as f:
                peptide_list = [line.strip() for line in f if line.strip()] # 增加一个判断，避免空行

            if peptide_list:
                peptides.extend(peptide_list)
                alleles.extend([allele] * len(peptide_list))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    Allele_peptide_dic = {'Allele':alleles,'Peptide_Sequence':peptides}
    df = pl.DataFrame(Allele_peptide_dic)
    df.write_csv('/data/qwu/data/TIPnovo/systemMHC/Allele_peptide.csv')


def search_files_in_subdir(subdir):
    """
    在单个子目录中递归搜索匹配 "*.mzML" 的文件。
    """
    try:
        # 确保输入是 Path 对象
        subdir_path = Path(subdir)
        # 使用 rglob 在子目录内进行递归搜索
        return list(subdir_path.rglob("*.mzML"))
    except Exception as e:
        print(f"Error processing directory {subdir}: {e}")
        return []

# 输入文件路径和目标谱图 ID
def getSpectrum(file_idList):
    """
    查找指定的谱图，并将其与对应的序列注释一起写入 MGF 文件。

    Args:
        file_idList (list): 一个包含四个元素的列表：
                            [0] mzML 文件路径 (str)
                            [1] 目标谱图 ID 列表 (list of str)
                            [2] 对应的肽段序列列表 (list of str)
                            [3] 输出 MGF 文件的路径 (str)
    """
    # 1. 从输入列表中解析参数
    file_path = file_idList[0]
    target_spectrum_id_list = file_idList[1]
    seq_list = file_idList[2]
    output_mgf_path = file_idList[3]

    # 2. 创建从谱图ID到序列的映射字典，并创建用于快速查找和追踪的集合
    if len(target_spectrum_id_list) != len(seq_list):
        print("Error: Spectrum ID list and Sequence list have different lengths.")
        return

    id_to_seq_map = {spec_id: seq for spec_id, seq in zip(target_spectrum_id_list, seq_list)}
    # 使用集合进行高效查找
    target_ids_to_find = set(target_spectrum_id_list)

    # 3. 打开 MGF 文件准备写入，并遍历 mzML 文件
    print(f"Starting to process {file_path}...")
    with open(output_mgf_path, 'w') as mgf_file:
        run = pymzml.run.Reader(file_path)
        for spectrum in run:
            current_id = spectrum.ID

            # 如果当前谱图是我们想要的
            if current_id in target_ids_to_find:

                # --- 提取 MGF 所需的元数据 ---
                # 对于MS2谱图，precursor信息通常是第一个
                precursor = spectrum.selected_precursors[0]
                charge = precursor.get('charge', 'N/A')
                pepmass = precursor.get('mz', 0.0)
                rt_in_seconds = spectrum.scan_time_in_minutes() * 60
                sequence = id_to_seq_map[current_id]

                # --- 写入 MGF 文件 ---
                mgf_file.write("BEGIN IONS\n")
                mgf_file.write(f"PEPMASS={pepmass}\n")
                mgf_file.write(f"CHARGE={charge}+\n")
                mgf_file.write(f"RTINSECONDS={rt_in_seconds}\n")
                mgf_file.write(f"SCANS={current_id}\n")  # 使用谱图ID作为SCANS
                mgf_file.write(f"SEQ={sequence}\n")

                # 提取并写入峰列表
                peaks = spectrum.peaks("centroided")

                # for mz, intensity in peaks:
                #     mgf_file.write(f"{mz} {intensity}\n")
                # 如果谱图没有峰，则直接结束当前条目
                if peaks.size == 0:
                    mgf_file.write("END IONS\n\n")
                    target_ids_to_find.remove(current_id)
                    if not target_ids_to_find: break
                    continue

                # 1. 查找这张谱图中的最高峰强度
                max_intensity = max(p[1] for p in peaks)

                # 2. 处理最高峰强度为0的边缘情况，防止除零错误
                if max_intensity == 0:
                    max_intensity = 1.0

                # 3. 遍历峰列表，写入m/z和归一化后的强度
                for mz, intensity in peaks:
                    # 计算相对强度
                    relative_intensity = intensity / max_intensity
                    mgf_file.write(f"{mz} {relative_intensity}\n")
                mgf_file.write("END IONS\n\n")

                # 从待查找集合中移除已找到的ID
                target_ids_to_find.remove(current_id)

                # 如果所有目标都已找到，提前结束循环
                if not target_ids_to_find:
                    break

    # 4. 循环结束后，检查是否所有谱图都已找到
    if target_ids_to_find:
        for not_found_id in target_ids_to_find:
            print(f"Warning: Could not find spectrum with ID {not_found_id} in {file_path}")

    print(f"Processing complete. MGF file saved to: {output_mgf_path}")

def parallel_file_search(base_path_str, main_pattern):
    """
    并行搜索文件。
    :param base_path_str: 基础搜索路径的字符串
    :param main_pattern: 用于确定要分发给各进程的子目录的模式 (e.g., "SYSMHC0*")
    """
    path_obj = Path(base_path_str)

    print("Finding all subdirectories to search...")
    # 1. 获取所有需要并行处理的子目录
    # os.scandir 通常比 Path.glob 更快，因为它返回迭代器且不立即获取所有属性
    subdirs_to_search = []
    with os.scandir(path_obj) as it:
        for entry in it:
            if entry.is_dir() and Path(entry.name).match(main_pattern):
                subdirs_to_search.append(entry.path)

    if not subdirs_to_search:
        print("No matching subdirectories found.")
        return []

    print(f"Found {len(subdirs_to_search)} subdirectories. Starting parallel search...")

    # 2. 创建进程池
    num_processes = 60
    print(f"Using {num_processes} processes.")

    all_found_files = []
    # 使用 with 语句可以确保进程池在使用后被正确关闭
    with Pool(processes=num_processes) as pool:
        # 3. 将搜索任务 (每个子目录) 映射到进程池
        results = pool.map(search_files_in_subdir, subdirs_to_search)

        # 4. 合并所有进程返回的结果
        for file_list in results:
            all_found_files.extend(file_list)

    return all_found_files

def pre_process():
    path = '/data/SYSTEMHC'
    SystemMHC_items_list = getSystem_dir(path)
    dataset_sample_dic = {}
    totall_data_size = 0 # kb
    for dataset in SystemMHC_items_list:
        try:
            sample_data_list,dataset_size = getSample_inDataset(os.path.join(path,dataset))
            if sample_data_list:
                dataset_sample_dic[dataset] = sample_data_list
                # print(dataset, dataset_size)
                totall_data_size += dataset_size
        except:
            pass

    # merged the psm list
    all_psm_path = '/data/qwu/data/TIPnovo/systemMHC'
    PSM_list = os.listdir('/data/qwu/data/TIPnovo/systemMHC')
    columns_list = []
    for i in PSM_list:
        df = pd.read_csv(os.path.join(all_psm_path,i))
        columns_list.append(df.columns.tolist())

    # polars
    lazy_frames = [
        pl.scan_csv(os.path.join(all_psm_path, i))
        .with_columns([
            pl.lit(i.split('___')[0]).alias('Dataset'),
            pl.lit(i.split('___')[1].split('.PSM')[0]).alias('sample')
        ])
        for i in PSM_list
    ]

    all_PSM_df = pl.concat(lazy_frames, how='diagonal').collect()

    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_all.PSM')
    df_filtered = df.filter(
        ~pl.col('Spectrum').str.contains('Spectrum')
    )
    # df_filtered.write_csv('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_all_filter2.PSM')

    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_all_filter.PSM')
    Massshift_type = df['Massshift'].value_counts()
    regex_pattern = r'([A-Z]\[\d+\])'
    df = df.with_columns(
        pl.col('Modified_Pep_Seq')  # 1. 选择要操作的列
        .str.extract(regex_pattern)  # 2. 提取修饰位点
        .fill_null('')  # 3. 将 null 值替换为空字符串 (这是正确的函数!)
        .alias('Modification_Site')  # 4. 给新生成的列命名
    )
    mod_valueCount = df['Modification_Site'].value_counts()
    Allele_peptide_df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Allele_peptide.csv')
    Allele_peptide_df = Allele_peptide_df.unique()
    alleles_agg = Allele_peptide_df.group_by('Peptide_Sequence').agg(
        pl.col('Allele').alias('Allele_List')  # .alias() 是给新列重命名
    )
    alleles_agg = alleles_agg.with_columns(
        pl.col('Allele_List').list.len().fill_null(0).alias('Allele_Count')
    )
    alleles_agg  = alleles_agg.with_columns(
        pl.col('Allele_List').list.join('___').fill_null('').alias('Allele_String')
    )
    alleles_agg = alleles_agg.drop('Allele_List')

    df = df.join(alleles_agg,on='Peptide_Sequence',how='left')
    # df = df.with_columns(
    #     pl.col('Allele_List').list.len().fill_null(0).alias('Allele_Count')
    # )
    # df = df.with_columns(
    #     pl.col('Allele_String').str.len().fill_null(0).alias('Allele_Count')
    # )
    Allele_count = df['Allele_Count'].value_counts()
    peptide_set = set(df['Peptide_Sequence'].to_list())
    df.write_csv('/data/qwu/data/TIPnovo/systemMHC/Allele_peptide_HLA.csv')

def PTM_Filter():
    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Allele_peptide_HLA.csv')
    PTM_valueCount = df['Modification_Site'].value_counts()
    df = df.drop_nulls(subset=['Allele_Count'])
    # binder 共有4093W PSM
    PTM_list = ['','M[147]','P[113]','W[202]','E[111]','Q[111]']
    df_filtered = df.filter(
        pl.col('Modification_Site').is_in(PTM_list)
    )
    df_filtered.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter.csv')
    # 符合条件 3922W

def Train_valid_test_split():
    df_filtered = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter.csv')
    df_filtered = df_filtered.with_columns(
        (pl.col('Peptide_Sequence') + "_"+pl.col('Charge').cast(pl.Utf8)).alias('Seq_charge')
    )
    df_filtered = df_filtered.filter(
        pl.col('iProbability') > 0.99
    )
    percursor_seq_list = df_filtered['Seq_charge'].unique().to_list()
    seq_list = df_filtered['Peptide_Sequence'].unique().to_list()
    ldf = df_filtered.lazy()

    # test
    df_Allele_1_peptide_list = df_filtered.filter(
        pl.col('Allele_Count') == 1
    )['Peptide_Sequence'].unique().to_list()
    test_list = random.sample(df_Allele_1_peptide_list, 60000) # 随机选择10000条
    test_df = df_filtered.filter(
        pl.col('Peptide_Sequence').is_in(test_list)
    )
    test_df = test_df.group_by("Peptide_Sequence").agg(
        pl.all().sort_by("iProbability", descending=True).head(2)
    )
    columns_to_explode = [col for col in test_df.columns if test_df.schema[col] == pl.List]
    test_df = test_df.explode(columns_to_explode)


    test_df.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_test.csv') # 10.7w PSM用于最后的验证，包含多个多价态

    df_filtered = df_filtered.filter(
        ~pl.col('Peptide_Sequence').is_in(test_list)
    )

    # 最后用于训练的调参的3249w PSM
    df_Allele_1_peptide_list = df_filtered.filter(
        pl.col('Allele_Count') == 1
    )['Peptide_Sequence'].unique().to_list()
    valid_list = random.sample(df_Allele_1_peptide_list, 60000) # 随机选择5000条
    valid_df = df_filtered.filter(
        pl.col('Peptide_Sequence').is_in(valid_list)
    )
    valid_df = valid_df.group_by("Peptide_Sequence").agg(
        pl.all().sort_by("iProbability", descending=True).head(2)
    )
    columns_to_explode = [col for col in valid_df.columns if valid_df.schema[col] == pl.List]
    valid_df = valid_df.explode(columns_to_explode)
    valid_df.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_valid.csv') # 10.7w PSM用于调参，包含多个多价态

    df_filtered = df_filtered.filter(
        ~pl.col('Peptide_Sequence').is_in(valid_list)
    )
    df_filtered.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_ALL.csv')

    # 最后完整的训练数据 3808w张
    # 40w peptide
    train_peptide_list = df_filtered['Peptide_Sequence'].unique().to_list()
    train_peptide_300000list = random.sample(train_peptide_list, 400000)
    df_300000filtered = df_filtered.filter(
        pl.col('Peptide_Sequence').is_in(train_peptide_300000list)
    )
    df_300000filtered = df_300000filtered.group_by("Peptide_Sequence").agg(
        pl.all().sort_by("iProbability", descending=True).head(3)
    )
    columns_to_explode = [col for col in df_300000filtered.columns if df_300000filtered.schema[col] == pl.List]
    df_300000filtered = df_300000filtered.explode(columns_to_explode)
    df_300000filtered.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_108w.csv')

    # 51w peptide
    train_peptide_list = df_filtered['Peptide_Sequence'].unique().to_list()
    df_ALLfiltered = df_filtered.filter(
        pl.col('Peptide_Sequence').is_in(train_peptide_list)
    )
    df_ALLfiltered = df_ALLfiltered.group_by("Peptide_Sequence").agg(
        pl.all().sort_by("iProbability", descending=True).head(2)
    )
    columns_to_explode = [col for col in df_ALLfiltered.columns if df_ALLfiltered.schema[col] == pl.List]
    df_ALLfiltered = df_ALLfiltered.explode(columns_to_explode)
    df_ALLfiltered.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_97w_AllPep.csv')

def getAnnoted_MGF(PSM_file):
    ###############################################################################################
    # =======================================================
    # == 请在这里修改为您真实的 `base_path` ==
    base_path = '/data/SYSTEMHC'
    # =======================================================

    print("--- Starting Parallel File Search ---")
    start_time = time.time()

    # main_pattern 对应 "SYSMHC0*"
    # rglob 在 worker 函数中处理剩下的 "*/*.mzML"
    all_peptide_files = parallel_file_search(base_path, "SYSMHC0*")

    end_time = time.time()

    print("\nFinished parallel search.")
    print(f"Found {len(all_peptide_files)} files in {end_time - start_time:.4f} seconds.")
    if all_peptide_files:
        print("First 5 files found:", all_peptide_files[:5])
    all_peptide_files = [x for x in all_peptide_files if 'Results' not in str(x)
                            and 'Comet' not in str(x) and
                            'Fragger' not in str(x) and
                            'MSGF' not in str(x) ]
    all_mzML_files_dic = dict(zip([x.name for x in all_peptide_files], [str(x) for x in all_peptide_files]))
    np.save('/data/qwu/data/TIPnovo/systemMHC/all_mzML_files_dic.npy',all_mzML_files_dic)

    all_mzML_files_dic = np.load('/data/qwu/data/TIPnovo/systemMHC/all_mzML_files_dic.npy',allow_pickle=True)[()]
    PSM_file = '/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_ALL.csv'
    df = pl.read_csv(PSM_file)
    df = df.with_columns(
        file=pl.col('Spectrum').str.split(by='.').list.get(0) + '.mzML'
    )
    df_filtered = df.filter(
        pl.col('file').is_in(all_mzML_files_dic.keys())
    )
    df_filtered = df_filtered.with_columns(
        mzML_path = pl.col('file').replace(all_mzML_files_dic)
    )
    df_filtered.write_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_3100WALL.csv')

def SystemMHC_tolerance():
    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_3100WALL.csv')
    Peptide_seq_list = df['Peptide_Sequence'].unique().to_list()
    precursor_list = df['Seq_charge'].unique().to_list()
    df_filtered = df.group_by("Seq_charge").agg(
        pl.all().sort_by("iProbability", descending=True).head(3)
    )
    columns_to_explode = [col for col in df_filtered.columns if df_filtered.schema[col] == pl.List]
    df_ALLfiltered = df_filtered.explode(columns_to_explode)
    df_ALLfiltered.write_parquet('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_200w_PSMReady.parquet')

def AnnoMGF_PSM(PSM_file):
    PSM_file = '/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_200w_PSMReady.parquet'
    df_filtered = pl.read_parquet(PSM_file)
    mzML_list = df_filtered['mzML_path'].unique().to_list()

    OUTPUT_DIR = '/data/qwu/data/TIPnovo/systemMHC/train_200wPSM'

    not_found_file_list = []
    #        file_idList (list): 一个包含四个元素的列表：
    #                         [0] mzML 文件路径 (str)
    #                         [1] 目标谱图 ID 列表 (list of str)
    #                         [2] 对应的肽段序列列表 (list of str)
    #                         [3] 输出 MGF 文件的路径 (str)
    for idx,mzML_path in enumerate(mzML_list):
        print(f"Annotating {idx} {mzML_path}")
        try:
            id_list = df_filtered.filter(pl.col('mzML_path') == mzML_path)['Scan_number'].to_list()
            sequence_list = df_filtered.filter(pl.col('mzML_path') == mzML_path)['Modified_Pep_Seq'].to_list()
            output_file = f"{OUTPUT_DIR}/{Path(mzML_path).stem}.mgf"
            getSpectrum([mzML_path,id_list,sequence_list,output_file])
        except:
            not_found_file_list.append(mzML_path)
            print(f"Error in {mzML_path}")

if __name__ == "__main__":
    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_3100WALL.csv')
    df = pl.read_csv('/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter.csv')
    mzML_path_list = list(set(df['mzML_path'].to_list()))
    analyzer_dic = get_analyzer_types_parallel(mzML_path_list)
    subprocess.run('rm /data/qwu/data/TIPnovo/model_save/test', shell=True)
    np.save('/data/qwu/data/TIPnovo/systemMHC/System_analyzer_dic.npy',analyzer_dic)
    # test = np.load('/data/qwu/data/TIPnovo/systemMHC/System_analyzer_dic.npy',allow_pickle=True)[()]
from pathlib import Path
import polars as pl
import multiprocessing
import os
import pymzml.run


# ==============================================================================
# 1. 核心函数 getSpectrum (保持不变)
# ==============================================================================
def getSpectrum(file_path, target_spectrum_id_list, seq_list, output_mgf_path):
    """
    查找指定的谱图，并将其与对应的序列注释一起写入 MGF 文件。
    此版本修复了函数签名，以直接接收多个参数。
    """
    try:
        # 如果没有要找的谱图，直接结束
        if not target_spectrum_id_list:
            return f"Skipped (no spectra): {Path(file_path).name}"

        # 确保映射字典的键和查找集合中的ID都是字符串
        id_to_seq_map = {str(spec_id): seq for spec_id, seq in zip(target_spectrum_id_list, seq_list)}
        target_ids_to_find = set(map(str, target_spectrum_id_list))

        with open(output_mgf_path, 'w') as mgf_file:
            run = pymzml.run.Reader(file_path)
            for spectrum in run:
                current_id = str(spectrum.ID)
                if current_id in target_ids_to_find:
                    precursor = spectrum.selected_precursors[0]
                    charge = precursor.get('charge', 'N/A')
                    pepmass = precursor.get('mz', 0.0)
                    rt_in_seconds = spectrum.scan_time_in_minutes() * 60
                    sequence = id_to_seq_map[current_id]

                    mgf_file.write("BEGIN IONS\n")
                    mgf_file.write(f"PEPMASS={pepmass}\n")
                    mgf_file.write(f"CHARGE={charge}+\n")
                    mgf_file.write(f"RTINSECONDS={rt_in_seconds}\n")
                    mgf_file.write(f"SCANS={current_id}\n")
                    mgf_file.write(f"SEQ={sequence}\n")

                    peaks = spectrum.peaks("centroided")
                    if peaks is not None and len(peaks) > 0:
                        max_intensity = max(p[1] for p in peaks)
                        if max_intensity == 0: max_intensity = 1.0
                        for mz, intensity in peaks:
                            relative_intensity = intensity / max_intensity
                            mgf_file.write(f"{mz} {relative_intensity}\n")
                    mgf_file.write("END IONS\n\n")

                    target_ids_to_find.remove(current_id)
                    if not target_ids_to_find:
                        break
        return f"Success: {Path(file_path).name}"
    except Exception as e:
        return f"!!! ERROR on {Path(file_path).name}: {e}"

# ==============================================================================
# 2. 主执行块 (这是主要修改部分)
# ==============================================================================
if __name__ == "__main__":
    # 设置 'spawn' 启动方式，以确保稳定性
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果上下文已设置，可能会抛出 RuntimeError，可以安全地忽略
        pass

    # 1. 设置参数
    PSM_FILE = '/data/qwu/data/TIPnovo/systemMHC/Binder_PTM_filter_train_3100WALL.csv'
    OUTPUT_DIR = Path('/data/qwu/data/TIPnovo/systemMHC/SystemMHC_ALLPSM')
    cpu_count_to_use = 15

    # 2. 准备数据
    print("Reading PSM file...")
    df_filtered = pl.read_csv(PSM_FILE)
    mzML_list = df_filtered['mzML_path'].unique().to_list()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ==================== 关键修改：在主进程中准备好所有任务参数 ====================
    print("Preparing tasks for parallel processing...")
    tasks = []
    for idx,mzml_path in enumerate(mzML_list):
        print(f"Preparing {idx+1}...")
        # 为每个 mzML 文件筛选出对应的数据
        df_subset = df_filtered.filter(pl.col('mzML_path') == mzml_path)


        # 准备 getSpectrum 函数所需的所有参数
        id_list = df_subset['Scan_number'].to_list()
        sequence_list = df_subset['Modified_Pep_Seq'].to_list()
        output_file = str(OUTPUT_DIR / f"{Path(mzml_path).stem}.mgf")

        # 将这组参数打包成一个元组，添加到任务列表
        tasks.append((mzml_path, id_list, sequence_list, output_file))
    # ================================================================================

    print(f"Starting to process {len(tasks)} tasks using {cpu_count_to_use} cores...")

    # 3. 创建并运行进程池
    # 使用 pool.starmap，它会自动解包每个任务元组并作为参数传给 getSpectrum
    with multiprocessing.Pool(processes=cpu_count_to_use) as pool:
        results = pool.starmap(getSpectrum, tasks)

    # (可选) 打印处理结果的摘要
    for res in results:
        if "ERROR" in res or "Skipped" in res:
            print(res)

    print("\nAll files processed.")
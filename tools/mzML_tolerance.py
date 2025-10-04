import os
import glob
from multiprocessing import Pool, cpu_count
from pyteomics import mzml
from tqdm import tqdm
import collections


def get_instrument_model(reader):
    """从 mzML 阅读器中提取仪器型号"""
    try:
        # 尝试从 referenceableParamGroupList 获取通用仪器参数
        # 这是 LTQ Orbitrap XL 等仪器存储信息的地方
        param_groups = reader.referenceable_param_groups
        if param_groups:
            for group_id, group in param_groups.items():
                # 寻找描述仪器型号的 cvParam
                for param in group:
                    if 'instrument model' in param or 'LTQ' in param or 'Orbitrap' in param or 'TOF' in param:
                        # 'name' 属性是 pyteomics v4.x 的用法
                        if hasattr(param, 'name'):
                            return param.name
                        # 兼容旧版 pyteomics 或不同文件格式
                        elif isinstance(param, dict):
                            return param.get('name', 'Unknown model')

        # 如果上面找不到，尝试从 instrumentConfiguration 中直接寻找
        configs = reader.instrument_configurations
        if configs:
            for config in configs:
                # 'componentList' 是 pyteomics v4.x 的用法
                if hasattr(config, 'componentList'):
                    # 也可以在这里寻找cvParam，但型号名通常在更高层
                    pass
                # 兼容旧版
                elif isinstance(config, dict) and 'cvParam' in config:
                    for param in config['cvParam']:
                        if 'instrument model' in param.get('name', ''):
                            return param.get('name')

        return "Unknown Instrument"
    except Exception:
        return "Error reading instrument model"


def analyze_resolution(reader, num_spectra_to_check=100):
    """分析谱图以确定 MS1 和 MS2 的分辨率配置"""
    ms_levels_config = collections.defaultdict(set)

    try:
        # 使用 iterfind 迭代器，避免一次性加载所有谱图索引
        count = 0
        for spectrum in reader:
            if count >= num_spectra_to_check:
                break

            ms_level = spectrum.get('ms level')

            # 查找 filter string，这是判断分辨率的关键
            scan = spectrum.get('scanList', {}).get('scan', [{}])[0]
            filter_string = scan.get('filter string', '').lower()

            if 'ftms' in filter_string:
                ms_levels_config[ms_level].add('High-Res (FTMS)')
            elif 'itms' in filter_string:
                ms_levels_config[ms_level].add('Low-Res (ITMS)')
            else:
                # 如果没有明确的标志，可以留空或记录部分 filter string
                ms_levels_config[ms_level].add('Unknown')

            count += 1

        if not ms_levels_config:
            return "No spectra found or metadata missing"

        # 格式化输出
        report = []
        for level, configs in sorted(ms_levels_config.items()):
            report.append(f"MS{level}: {', '.join(sorted(list(configs)))}")
        return "; ".join(report)

    except Exception as e:
        return f"Error analyzing spectra: {e}"


def process_mzml_file(filepath):
    """
    工作函数：处理单个 mzML 文件。
    """
    try:
        with mzml.MzML(filepath) as reader:
            instrument_model = get_instrument_model(reader)
            resolution_config = analyze_resolution(reader)

            return {
                "file": os.path.basename(filepath),
                "instrument_model": instrument_model,
                "resolution_config": resolution_config
            }
    except Exception as e:
        return {
            "file": os.path.basename(filepath),
            "error": str(e)
        }



if __name__ == '__main__':
    import os
    import glob
    from multiprocessing import Pool, cpu_count
    from pyteomics import mzml
    from tqdm import tqdm
    import collections


    def get_instrument_model(reader):
        """从 mzML 阅读器中提取仪器型号"""
        try:
            # 尝试从 referenceableParamGroupList 获取通用仪器参数
            # 这是 LTQ Orbitrap XL 等仪器存储信息的地方
            param_groups = reader.referenceable_param_groups
            if param_groups:
                for group_id, group in param_groups.items():
                    # 寻找描述仪器型号的 cvParam
                    for param in group:
                        if 'instrument model' in param or 'LTQ' in param or 'Orbitrap' in param or 'TOF' in param:
                            # 'name' 属性是 pyteomics v4.x 的用法
                            if hasattr(param, 'name'):
                                return param.name
                            # 兼容旧版 pyteomics 或不同文件格式
                            elif isinstance(param, dict):
                                return param.get('name', 'Unknown model')

            # 如果上面找不到，尝试从 instrumentConfiguration 中直接寻找
            configs = reader.instrument_configurations
            if configs:
                for config in configs:
                    # 'componentList' 是 pyteomics v4.x 的用法
                    if hasattr(config, 'componentList'):
                        # 也可以在这里寻找cvParam，但型号名通常在更高层
                        pass
                    # 兼容旧版
                    elif isinstance(config, dict) and 'cvParam' in config:
                        for param in config['cvParam']:
                            if 'instrument model' in param.get('name', ''):
                                return param.get('name')

            return "Unknown Instrument"
        except Exception:
            return "Error reading instrument model"


    def analyze_resolution(reader, num_spectra_to_check=100):
        """分析谱图以确定 MS1 和 MS2 的分辨率配置"""
        ms_levels_config = collections.defaultdict(set)

        try:
            # 使用 iterfind 迭代器，避免一次性加载所有谱图索引
            count = 0
            for spectrum in reader:
                if count >= num_spectra_to_check:
                    break

                ms_level = spectrum.get('ms level')

                # 查找 filter string，这是判断分辨率的关键
                scan = spectrum.get('scanList', {}).get('scan', [{}])[0]
                filter_string = scan.get('filter string', '').lower()

                if 'ftms' in filter_string:
                    ms_levels_config[ms_level].add('High-Res (FTMS)')
                elif 'itms' in filter_string:
                    ms_levels_config[ms_level].add('Low-Res (ITMS)')
                else:
                    # 如果没有明确的标志，可以留空或记录部分 filter string
                    ms_levels_config[ms_level].add('Unknown')

                count += 1

            if not ms_levels_config:
                return "No spectra found or metadata missing"

            # 格式化输出
            report = []
            for level, configs in sorted(ms_levels_config.items()):
                report.append(f"MS{level}: {', '.join(sorted(list(configs)))}")
            return "; ".join(report)

        except Exception as e:
            return f"Error analyzing spectra: {e}"


    def process_mzml_file(filepath):
        """
        工作函数：处理单个 mzML 文件。
        """
        try:
            with mzml.MzML(filepath) as reader:
                instrument_model = get_instrument_model(reader)
                resolution_config = analyze_resolution(reader)

                return {
                    "file": os.path.basename(filepath),
                    "instrument_model": instrument_model,
                    "resolution_config": resolution_config
                }
        except Exception as e:
            return {
                "file": os.path.basename(filepath),
                "error": str(e)
            }


    def main():
        """
        主函数：设置多进程并处理所有 mzML 文件。
        """
        # =========================================================================
        #  !!! 请修改这里为你存放 mzML 文件的文件夹路径 !!!
        #  可以使用通配符 * 来匹配所有 .mzML 或 .mzml 文件
        # =========================================================================
        search_path = "./data/*.mzML"  # 示例路径，请根据实际情况修改

        mzml_files = glob.glob(search_path)

        if not mzml_files:
            print(f"在路径 '{search_path}' 下没有找到任何 mzML 文件。请检查路径。")
            return

        print(f"找到 {len(mzml_files)} 个 mzML 文件，开始处理...")

        # 使用所有可用的 CPU 核心，也可以指定数量，如 Pool(4)
        num_processes = min(cpu_count(), len(mzml_files))

        results = []
        with Pool(processes=num_processes) as pool:
            # 使用 tqdm 显示进度条
            # imap_unordered 可以让完成的任务立刻被主进程获取，进度条更新更平滑
            with tqdm(total=len(mzml_files), desc="Processing mzML files") as pbar:
                for result in pool.imap_unordered(process_mzml_file, mzml_files):
                    results.append(result)
                    pbar.update(1)

        # 打印结果
        print("\n--- 分析结果 ---")
        # 为了保持输出顺序，可以根据文件名对结果进行排序
        results.sort(key=lambda x: x['file'])

        for res in results:
            if "error" in res:
                print(f"文件: {res['file']}")
                print(f"  - 错误: {res['error']}")
            else:
                print(f"文件: {res['file']}")
                print(f"  - 仪器型号: {res['instrument_model']}")
                print(f"  - m/z 采集配置: {res['resolution_config']}")
            print("-" * 20)


    if __name__ == '__main__':
        main()
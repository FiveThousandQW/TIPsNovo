import pandas as pd
import polars as pl


def pred_target(pred_result):
    # proteometools part1 40w
    pred_df = pd.read_csv(pred_result)
    pred_seq_list = [x.replace('M[UNIMOD:35]','M[147]') for x in pred_df['preds'].to_list()]
    target_df = pd.read_csv('/data/48/wuqian/fast/soft/casanovo/code/test_instanovo/tests/SystemMHC_valid_6w_extracted_data.csv')
    target_df['preds'] = pred_seq_list
    df = target_df
    df = df[~df['SEQ'].str.contains('\[')]

    df['SEQ'] = df['SEQ'].apply(lambda x: x.replace('I','L'))
    df['preds'] = df['preds'].apply(lambda x: x.replace('I', 'L'))
    df['len_SEQ'] = df['SEQ'].apply(
        lambda x: len(x) if '[' not in x else len(x) -5
    )
    df['len_preds'] = df['preds'].apply(
        lambda x: len(x) if '[' not in x else len(x) -5
    )
    df['match'] = df['SEQ'] == df['preds']
    df['len_match'] = df['len_SEQ'] == df['len_preds']
    df['match'].value_counts()
    pep_recall = (df['match'].value_counts()[True]) / (df['match'].value_counts()[True] + df['match'].value_counts()[False])
    print(pep_recall)


if __name__ == '__main__':
    path1 = '/home/qwu/soft/casanovo/code/test_instanovo/tests/FineTune_SystemMHC_6wValid_result'
    path2 = '/home/qwu/soft/casanovo/code/test_instanovo/tests/Original_weight_stemMHC_6wValid_result'
    pred_target(path1)

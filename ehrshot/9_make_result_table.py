import os
import argparse
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from utils import (
    LABELING_FUNCTIONS, 
    TASK_GROUP_2_LABELING_FUNCTION, 
    SHOT_STRATS,
    filter_df,
    get_rel_path,
    type_tuple_list,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Make table ")
    parser.add_argument("--path_to_results_dir", default=get_rel_path(__file__, '../EHRSHOT_ASSETS/outputs/'), type=str, help="Path to directory containing results from 6_eval.py")
    parser.add_argument("--shot_strat", required=True, type=str, choices=SHOT_STRATS.keys(), help="What type of k-shot evaluation we are interested in.")
    parser.add_argument("--model_heads", type=type_tuple_list, required=True, help="Specific (model, head) combinations to plot. Format it as a Python list of tuples of strings, e.g. [('clmbr', 'lr_lbfgs'), ('count', 'gbm')]")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_RESULTS_DIR: str = args.path_to_results_dir
    SHOT_STRAT: str = args.shot_strat
    MODEL_HEADS: List[Tuple[str, str]] = args.model_heads
    
    # Load all results from CSVs
    dfs: List[pd.DataFrame] = []
    for idx, labeling_function in tqdm(enumerate(LABELING_FUNCTIONS)):
        path_to_csv: str = os.path.join(PATH_TO_RESULTS_DIR, f"{labeling_function}/{SHOT_STRAT}_results.csv")
        if os.path.exists(path_to_csv):
            dfs.append(pd.read_csv(path_to_csv))
        else:
            print(f"Skipping: {labeling_function} b/c no file at `{path_to_csv}`")
    df_results: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    header = ['model']
    for task_group, funcs in TASK_GROUP_2_LABELING_FUNCTION.items():
        if task_group == 'chexpert':
            continue # disabled for now
        header.append(task_group)

    print("Average AUROC For Each Task Group")

    print('|' + ' | '.join(header) + '|')

    print('|' + ' | '.join('--' for _ in header) + '|')

    for model_head in MODEL_HEADS:
        row = [model_head[0] + ' with ' + model_head[1]]
        for task_group, funcs in TASK_GROUP_2_LABELING_FUNCTION.items():
            if task_group == 'chexpert':
                continue # disabled for now
            aurocs = []
            for func in funcs:
                filtered_df = filter_df(df_results, score='auroc', labeling_function=func, model_heads=[model_head], k=-1)
                assert len(filtered_df) == 1, f'{len(filtered_df)} {filtered_df} {func}'
                aurocs.append(list(filtered_df['value'])[0])
            row.append(f'{sum(aurocs) / len(aurocs):0.3f}')
        print('|' + ' | '.join(row) + '|')
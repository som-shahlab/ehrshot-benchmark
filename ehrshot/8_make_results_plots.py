import os
import argparse
from typing import List, Optional, Tuple, Set
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    HEAD_2_INFO,
    MODEL_2_INFO, 
    SHOT_STRATS,
    filter_df,
)
from ehrshot.plot import (
    plot_one_labeling_function,
    plot_one_task_group,
    plot_one_task_group_box_plot,
    _plot_unified_legend,
    map_model_head_to_color,
)

def plot_individual_tasks(df_results: pd.DataFrame, 
                                score: str, 
                                path_to_output_dir: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None,
                                is_x_scale_log: bool = True,
                                is_std_bars: bool = True):
    labeling_functions: List[str] = df_results[df_results['score'] == score]['labeling_function'].unique().tolist()
    fig, axes = plt.subplots(int(np.ceil(len(labeling_functions) / 3)), 3, figsize=(20, 20))
    for idx, labeling_function in enumerate(labeling_functions):
        sub_tasks: List[str] = df_results[(df_results['score'] == score) & (df_results['labeling_function'] == labeling_function)]['sub_task'].unique().tolist()
        plot_one_labeling_function(df_results, 
                                    axes.flat[idx], 
                                    labeling_function, 
                                    sub_tasks, 
                                    score,
                                    model_heads=model_heads,
                                    is_x_scale_log=is_x_scale_log,
                                    is_std_bars=False if labeling_function == 'chexpert' else is_std_bars)

    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, ncol=4, fontsize=8)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.savefig(os.path.join(path_to_output_dir, f"tasks_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_taskgroups(df_results: pd.DataFrame, 
                        score: str, 
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None,
                        is_x_scale_log: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in enumerate(task_groups):
        plot_one_task_group(df_results, 
                            axes.flat[idx], 
                            task_group, 
                            score,
                            model_heads=model_heads,
                            is_x_scale_log=is_x_scale_log)
    
    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, ncol=4, fontsize=8)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_taskgroups_box_plots(df_results: pd.DataFrame,
                            score: str, 
                            path_to_output_dir: str,
                            model_heads: Optional[List[Tuple[str, str]]] = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in tqdm(enumerate(task_groups)):
        plot_one_task_group_box_plot(df_results, 
                                    axes.flat[idx], 
                                    task_group, 
                                    score,
                                    model_heads=model_heads)
    
    # Create a unified legend for the entire figure
    df_ = filter_df(df_results, score=score, model_heads=model_heads)
    legend_n_col: int = 4
    all_model_heads = [ (x,y) for x,y in df_[['model', 'head']].drop_duplicates().itertuples(index=False) ]
    handles = [
        Patch(
            facecolor=map_model_head_to_color(model, head, all_model_heads),
            edgecolor=map_model_head_to_color(model, head, all_model_heads),
            label=f"{MODEL_2_INFO[model]['label']}+{HEAD_2_INFO[head]['label']}"
        ) 
        for (model, head) in all_model_heads
    ]
    fig.legend(handles=handles, loc='lower center', ncol=legend_n_col, fontsize=12)
    
    # Plot aesthetics
    fig.suptitle(f'Few-shot v. Full data {score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_boxplot_{score}.png"), dpi=300)
    plt.close('all')
    return fig


def merge_html_tables(path_to_output_dir: str, score: str):
    # Merge together all .html tables for easy copying
    contents = []
    for file in os.listdir(path_to_output_dir):
        if file.endswith('_pretty.html'):
            name = file.replace('_pretty.html', '')
            name = LABELING_FUNCTION_2_PAPER_NAME[name] if name in LABELING_FUNCTION_2_PAPER_NAME else (TASK_GROUP_2_PAPER_NAME[name] if name in TASK_GROUP_2_PAPER_NAME else name)
            table = open(os.path.join(path_to_output_dir, file), 'r').read()
            table = table.replace('\n', '')
            contents.append(f'<h5>{name}</h5>\n' + table)
    with open(os.path.join(path_to_output_dir, '../', score + '_merged.html'), 'w') as fd:
        for c in contents:
            fd.write(c + '\n\n')

def merge_md_tables(path_to_output_dir: str, score: str):
    # Merge together all .md tables for easy copying
    contents = []
    for file in os.listdir(path_to_output_dir):
        if file.endswith('_pretty_all.md'):
            name = file.replace('_pretty_all.md', '')
            name = LABELING_FUNCTION_2_PAPER_NAME[name] if name in LABELING_FUNCTION_2_PAPER_NAME else (TASK_GROUP_2_PAPER_NAME[name] if name in TASK_GROUP_2_PAPER_NAME else name)
            table = open(os.path.join(path_to_output_dir, file), 'r').read()
            contents.append(f'# {name}\n' + table + '\n\n')
    with open(os.path.join(path_to_output_dir, '../', score + '_merged_all.md'), 'w') as fd:
        for c in contents:
            fd.write(c + '\n\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Make plots of results")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to directory containing saved labels and featurizers")
    parser.add_argument("--path_to_results_dir", required=True, type=str, help="Path to directory containing results from 7_eval.py")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory to save figures")
    parser.add_argument("--shot_strat", required=True, type=str, choices=SHOT_STRATS.keys(), help="What type of k-shot evaluation we are interested in.")
    parser.add_argument("--model_heads", type=str, default=None, help="Specific (model, head) combinations to plot. Format it as a Python list of tuples of strings, e.g. [('clmbr', 'lr'), ('count', 'gbm')]")
    parser.add_argument("--is_skip_tables", action="store_true", default=False, help="If TRUE, then skip creating tables")
    parser.add_argument("--is_skip_plots", action="store_true", default=False, help="If TRUE, then skip creating plots")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_RESULTS_DIR: str = args.path_to_results_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    SHOT_STRAT: str = args.shot_strat
    MODEL_HEADS: Optional[List[Tuple[str, str]]] = eval(args.model_heads) if args.model_heads is not None else None
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    
    # Load all results from CSVs
    dfs: List[pd.DataFrame] = []
    for idx, labeling_function in tqdm(enumerate(LABELING_FUNCTION_2_PAPER_NAME.keys())):
        path_to_csv = os.path.join(PATH_TO_RESULTS_DIR, f"{labeling_function}/{SHOT_STRAT}_results.csv")
        if not os.path.exists(path_to_csv): 
            print("Skipping ", labeling_function)
            continue
        dfs.append(pd.read_csv(path_to_csv))
    df_results: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    
    # Start by filtering out MODEL_HEADS (we do this for all plots, so do it upfront)
    if MODEL_HEADS is not None:
        df_results = filter_df(df_results, model_heads=MODEL_HEADS)

    # TODO - drop brier for now
    df_results = df_results[df_results['score'] != 'brier']
    
    # Check with (model,head,subtask,score,k) combinations are missing
    all_combinations: Set[Tuple] = set(list(df_results[['model', 'head', 'sub_task', 'score', 'k']].drop_duplicates().itertuples(index=False, name=None)))
    missing_combinations = []
    for model, head in MODEL_HEADS:
        for sub_task in df_results['sub_task'].unique():
            for score in df_results['score'].unique():
                for k in df_results['k'].unique():
                    if (model, head, sub_task, score, k) not in all_combinations:
                        missing_combinations.append((model, head, sub_task, score, k))
    expected_combinations: int = len(MODEL_HEADS) * len(df_results['sub_task'].unique()) * len(df_results['score'].unique()) * len(df_results['k'].unique())
    assert expected_combinations == len(all_combinations) + len(missing_combinations), f"Expected {expected_combinations} total combinations, but found in df_results: {len(all_combinations)} (actual) + {len(missing_combinations)} (missing) = {len(all_combinations) + len(missing_combinations)}"
    if len(missing_combinations) > 0:
        print("==========================")
        print(f"You are missing a total of `{len(missing_combinations)}` combinations of (model, head, sub_task, score, k). Try rerunning `7_eval.sh` to generate these results.")
        print(f"The missing combinations are:")
        if len(missing_combinations) >= 100:
            df_missing = pd.DataFrame(missing_combinations, columns=['model', 'head', 'sub_task', 'score', 'k'])
            print("\nBy model:")
            print(df_missing.groupby(['model']).agg({'score' : 'first', 'head' : 'first', 'sub_task' : 'first', 'k' : 'count'}).reset_index().drop(columns=['head', 'sub_task', 'score']).sort_values(by=['k', 'model', ], ascending=False))
            print("\nBy model + head:")
            print(df_missing.groupby(['model', 'head']).agg({'score' : 'first', 'sub_task' : 'first', 'k' : 'count'}).reset_index().drop(columns=['sub_task', 'score']).sort_values(by=['k', 'model', 'head',], ascending=False))
            print("\nBy model + head + task:")
            print(df_missing.groupby(['model', 'head', 'sub_task',]).agg({'score' : 'first', 'k' : 'count'}).reset_index().drop(columns=['score']).sort_values(by=['k' ,'model', 'head',], ascending=False))
        else:
            for m in missing_combinations:
                print("\t", m)
        print("==========================")

    ####################################
    ####################################
    #
    # Tables
    #
    ####################################
    ####################################
    if not args.is_skip_tables:
        df_means = df_results.groupby([
            'labeling_function',
            'sub_task',
            'model',
            'head',
            'score',
            'k',
        ]).agg({
            'value' : 'mean',
            'k' : 'first',
            'labeling_function' : 'first',
            'sub_task' : 'first',
            'model' : 'first',
            'head' : 'first',
            'score' : 'first',
        }).reset_index(drop = True)
        df_stds = df_results.groupby([
            'labeling_function',
            'sub_task',
            'model',
            'head',
            'score',
            'k',
        ]).agg({
            'value' : 'std',
            'k' : 'first',
            'labeling_function' : 'first',
            'sub_task' : 'first',
            'model' : 'first',
            'head' : 'first',
            'score' : 'first',
        }).reset_index(drop = True).fillna(0)
        df_means['k'] = df_means['k'].astype(int)
        df_stds['k'] = df_stds['k'].astype(int)
        
        # Table for each (labeling function, score)
        #   Rows = model + head
        #   Columns = k
        #   Cells = mean ± std of score
        for score in tqdm(df_means['score'].unique(), desc='tables_individual_tasks()'):
            path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'individual_tasks', score)
            for sub_task in df_means['sub_task'].unique():
                os.makedirs(path_to_output_dir_, exist_ok=True)
                df_ = filter_df(df_means, sub_tasks=[sub_task], score=score, model_heads=MODEL_HEADS).sort_values(by=['model', 'head', 'k'])

                if df_.shape[0] == 0:
                    # No rows found for this MODEL_HEADS, so skip
                    print(f"No rows found for `{sub_task}` in `{score}` for `{MODEL_HEADS}`")
                    continue

                df_ = df_.rename(columns = {'value' : 'mean' })
                df_std_ = df_stds[(df_stds['sub_task'] == sub_task) & (df_stds['score'] == score)].sort_values(by=['model', 'head', 'k'])
                df_['std'] = df_std_['value']
                # Save raw df
                df_.to_csv(os.path.join(path_to_output_dir_, f'{sub_task}_raw.csv'), index=False)
                # Save pretty df
                df_ = df_.drop(columns = ['score', 'sub_task', 'labeling_function'])
                df_['value'] = df_['mean'].round(3).astype(str) + ' ± ' + df_['std'].round(3).astype(str)
                df_ = df_.drop(columns=['mean', 'std'])
                df_ = df_.pivot(index=['model', 'head'], columns='k', values='value').reset_index()
                df_.columns = [ str(x) for x in df_.columns ]
                df_ = df_.rename(columns={'-1' : 'All', '-1.0' : 'All'})
                df_.to_csv(os.path.join(path_to_output_dir_, f'{sub_task}_pretty.csv'), index=False)
                # Create Markdown table with just `All`
                df_all_ = df_[['model', 'head', 'All']].sort_values(['All'], ascending=False)
                df_all_.to_markdown(os.path.join(path_to_output_dir_, f'{sub_task}_pretty_all.md'), index=False)
                # Create HTML Table with multicolumn header
                df_['model'] = df_['model'] + ' - ' + df_['head']
                df_ = df_.drop(columns=['head'])
                df_.columns = pd.MultiIndex.from_tuples([
                    ('Model', ''),
                    ('All', ''),
                ] + [ ('K', x) for x in df_.columns[2:] ])
                df_.to_html(os.path.join(path_to_output_dir_, f'{sub_task}_pretty.html'), classes=['leaderboard_table'], index=False)
            # Merge together all HTML tables for easy copying
            merge_html_tables(path_to_output_dir_, score)
            merge_md_tables(path_to_output_dir_, score)

        # Table for each (task group, score)
        #   Rows = model + head
        #   Columns = k
        #   Cells = mean ± std of score
        task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
        for score in tqdm(df_means['score'].unique(), desc='tables_task_groups()'):
            path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'task_groups', score)
            for task_group in task_groups:
                os.makedirs(path_to_output_dir_, exist_ok=True)
                df_ = filter_df(df_means, task_group=task_group, score=score, model_heads=MODEL_HEADS)
                
                if df_.shape[0] == 0:
                    # No rows found for this MODEL_HEADS, so skip
                    print(f"No rows found for `{sub_task}` in `{score}` for `{MODEL_HEADS}`")
                    continue
                
                # Do another round of averaging over all subtasks:
                df_ = df_.groupby([
                    'model',
                    'head',
                    'k',
                ]).agg({
                    'value' : 'mean',
                    'k' : 'first',
                    'labeling_function' : 'first',
                    'sub_task' : 'first',
                    'model' : 'first',
                    'head' : 'first',
                    'score' : 'first'
                }).reset_index(drop = True)
                df_ = df_.rename(columns = {'value' : 'mean' })
                # Save raw df
                df_.to_csv(os.path.join(path_to_output_dir_, f'{task_group}_raw.csv'), index=False)
                # Save pretty df
                df_ = df_.drop(columns = ['score', 'sub_task', 'labeling_function'])
                df_['value'] = df_['mean'].round(3).astype(str)
                df_ = df_.drop(columns=['mean', ])
                df_ = df_.pivot(index=['model', 'head'], columns='k', values='value').reset_index()
                df_.columns = [ str(x) for x in df_.columns ]
                df_ = df_.rename(columns={'-1' : 'All'})
                df_.to_csv(os.path.join(path_to_output_dir_, f'{task_group}_pretty.csv'), index=False)
                # Create Markdown table with just `All`
                df_all_ = df_[['model', 'head', 'All']].sort_values(['All'], ascending=False)
                df_all_.to_markdown(os.path.join(path_to_output_dir_, f'{task_group}_pretty_all.md'), index=False)
                # Create HTML Table with multicolumn header
                df_['model'] = df_['model'] + ' - ' + df_['head']
                df_ = df_.drop(columns=['head'])
                df_.columns = pd.MultiIndex.from_tuples([
                    ('Model', ''),
                    ('All', ''),
                ] + [ ('K', x) for x in df_.columns[2:] ])
                df_.to_html(os.path.join(path_to_output_dir_, f'{task_group}_pretty.html'), classes=['leaderboard_table'], index=False)
            # Merge together all HTML tables for easy copying
            merge_html_tables(path_to_output_dir_, score)
            merge_md_tables(path_to_output_dir_, score)
            
    ####################################
    ####################################
    #
    # Plots
    #
    ####################################
    ####################################
    if not args.is_skip_plots:
        # Plotting aggregated auroc and auprc plots by task groups
        for score in tqdm(df_results['score'].unique(), desc='plot_taskgroups()'):
            if score == 'brier': continue
            plot_taskgroups(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR, 
                                model_heads=MODEL_HEADS, is_x_scale_log=True)

        # Plotting individual AUROC/AUPRC plot for each labeling function
        for score in tqdm(df_results['score'].unique(), desc='plot_individual_tasks()'):
            if score == 'brier': continue
            plot_individual_tasks(df_results, score, PATH_TO_OUTPUT_DIR, 
                                        model_heads=MODEL_HEADS, is_x_scale_log=True, is_std_bars=True)

        # plotting aggregated auroc and auprc box plots by task groups
        # for score in tqdm(df_results['score'].unique(), desc='plot_taskgroups_box_plots()'):
        #     if score == 'brier': continue
        #     plot_taskgroups_box_plots(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR,
        #                                 model_heads=MODEL_HEADS)
            
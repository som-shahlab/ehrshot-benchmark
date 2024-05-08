import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    HEAD_2_NAME,
    MODEL_2_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    SCORE_MODEL_HEAD_2_COLOR,
    filter_df,
)

def _plot_unified_legend(fig, axes, ncol=None, fontsize=14):
    """Create a unified legend for the entire figure."""
    labels = []
    label2handle = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                label2handle[l] = h
    legend_n_col: int = len([ x for x in labels if '(Full)' not in x ]) if ncol is None else ncol
    fig.legend([ label2handle[l] for l in labels ], labels, loc='lower center', ncol=legend_n_col, fontsize=fontsize)

def plot_one_labeling_function(df: pd.DataFrame,
                                ax: plt.Axes,
                                labeling_function: str,
                                sub_tasks: List[str],
                                score: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None,
                                is_x_scale_log: bool = True,
                                is_std_bars: bool = True,
                                path_to_output_table: str = None):
    """
        Graph: Line plot of each model+head's results for a single labeling function as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            lines = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
    """
    # Limit to specific labeling_function, subtask, score, (model, head) combos
    df = filter_df(df, score=score, labeling_function=labeling_function, sub_tasks=sub_tasks, model_heads=model_heads)

    if labeling_function == 'new_celiac':
        # Only 62 train examples, so cutoff plot at `k = 64`
        df = df[df['k'] <= 64]

    # Get all `k` shots tested
    ks: List[int] = sorted(df['k'].unique().tolist())
    
    # Create a fake `k` for the full data which is 2x the max `k` in the few-shot data
    x_tick_labels: List[str] = ks
    if -1 in ks:
        ks.remove(-1)
        full_data_k: int = 2 * max(ks)
        ks.append(full_data_k)
        df.loc[df['k'] == -1, 'k'] = full_data_k
        x_tick_labels = [ str(k) if k != full_data_k else 'All' for k in ks ]
    
    df_means = df.groupby([
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
    df_stds = df.groupby([
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
    }).reset_index(drop = True)

    models: List[str] = df['model'].unique().tolist()
    df_table = [] # Save what's actually plotted as a CSV
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            model_name: str = MODEL_2_NAME[model]
            head_name: str = HEAD_2_NAME[head]

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')
            df_stds_ = df_stds[(df_stds['model'] == model) & (df_stds['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual subtasks
            for subtask in df_means_['sub_task'].unique():
                df_m_ = df_means_[df_means_['sub_task'] == subtask]
                df_s_ = df_stds_[df_stds_['sub_task'] == subtask]
                ax.plot(df_m_['k'], df_m_['value'], color=color, linestyle='-', linewidth=2, alpha=0.25)
                if is_std_bars:
                    ax.plot(df_m_['k'], df_m_['value'] - df_s_['value'], color=color, alpha=0.1)
                    ax.plot(df_m_['k'], df_m_['value'] + df_s_['value'], color=color, alpha=0.1)
                    ax.fill_between(df_m_['k'], df_m_['value'] - df_s_['value'], df_m_['value'] + df_s_['value'],color=color, alpha=0.2)

            # Plot average line across all subtasks
            df_ = df_means_.groupby(['k']).agg({ 'value' : 'mean', 'k': 'first', }).reset_index(drop = True)
            ax.plot(df_['k'], df_['value'], color=color, label=f'{model_name}+{head_name}', linestyle='-', marker='o', linewidth=3, markersize=7)
            df_table.append(df_)

    # Plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(LABELING_FUNCTION_2_PAPER_NAME[labeling_function], size=14)
    ax.set_ylabel(score.upper(), fontsize=10)
    ax.set_xlabel("# of Train Examples per Class", fontsize=10)
    ax.set_xticks(ks, ks)
    ax.set_xticklabels(x_tick_labels)
    
    # Dump table
    if path_to_output_table and len(df_table) > 0:
        df_table = pd.concat(df_table)
        df_table.to_csv(path_to_output_table, index=False)


    
def plot_one_task_group(df: pd.DataFrame, 
                        ax: plt.Axes, 
                        task_group: str, 
                        score: str, 
                        model_heads: Optional[List[Tuple[str, str]]] = None, 
                        is_x_scale_log: bool = True,
                        path_to_output_table: str = None):    
    """
        Graph: Aggregated line plot of each model+head's results for all of the labeling functions within a task group, as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            dark lines = model+head's achieved mean score across replicates, averaged across all labeling functions in this task group
            faded lines = model+head's achieved mean score across replicates for each individual labeling function
    """

    # Limit to specific task_group, score, (model, head) combos
    df = filter_df(df, score=score, task_group=task_group, model_heads=model_heads)

    # Get all `k` shots tested
    ks: List[int] = sorted(df['k'].unique().tolist())
    
    # Create a fake `k` for the full data which is 2x the max `k` in the few-shot data
    x_tick_labels: List[str] = ks
    if -1 in ks:
        ks.remove(-1)
        full_data_k: int = 2 * max(ks)
        ks.append(full_data_k)
        df.loc[df['k'] == -1, 'k'] = full_data_k
        x_tick_labels = [ str(k) if k != full_data_k else 'All' for k in ks ]

    df_means = df.groupby([
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

    models: List[str] = df['model'].unique().tolist()
    df_table = [] # Save what's actually plotted as a CSV
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            model_name: str = MODEL_2_NAME[model]
            head_name: str = HEAD_2_NAME[head]

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual subtasks
            for subtask in df_means_['sub_task'].unique():
                df_m_ = df_means_[df_means_['sub_task'] == subtask]
                ax.plot(df_m_['k'], df_m_['value'], color=color, linestyle='-', linewidth=2, alpha=0.25)
    
            # Plot average line per model
            df_ = df_means_.groupby(['k']).agg({ 'value' : 'mean', 'k': 'first', }).reset_index(drop = True)
            ax.plot(df_['k'], df_['value'], color=color, label=f'{model_name}+{head_name}', linestyle='-', marker='o', linewidth=3, markersize=7)
            df_table.append(df_)

    # Plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(TASK_GROUP_2_PAPER_NAME[task_group], size=14)
    ax.set_ylabel(score.upper(), fontsize=10)
    ax.set_xlabel("# of Train Examples per Class", fontsize=10)
    ax.set_xticks(ks, ks)
    ax.set_xticklabels(x_tick_labels)
    
    # Dump table
    if path_to_output_table and len(df_table) > 0:
        df_table = pd.concat(df_table)
        df_table = df_table.rename(columns={'value' : 'mean'})
        df_table.to_csv(path_to_output_table, index=False)


def plot_one_task_group_box_plot(df: pd.DataFrame, 
                                ax: plt.Axes,
                                task_group: str, 
                                score: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None):
    """
        Graph: Aggregated box plot containing each model+head's results for all of the labeling functions within a task group, as a function of `k`,
            where results are the relative difference between the `k`-shot model+head and the full data model+head.
    
            y-axis = model+head's achieved score across replicates and labeling functions within a task group, relative to its score with full data
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    """
    # Select specific task group, score, (model, head) combos
    df = filter_df(df, task_group=task_group, score=score, model_heads=model_heads)
    
    # Get all `k` shots tested
    ks: List[int] = sorted(df['k'].unique().tolist())
    
    # Create a fake `k` for the full data which is 2x the max `k` in the few-shot data
    assert -1 in ks, f"Full data not present in {task_group} for {score}"
    ks.remove(-1)
    full_data_k: int = 2 * max(ks)
    df.loc[df['k'] == -1, 'k'] = full_data_k
    x_tick_labels = ks
    positions: np.ndarray = np.arange(len(ks))

    # Merge all scores at each `k` across all tasks for each model+head
    df_grouped = df.groupby([
        'model',
        'head',
        'score',
        'k',
    ]).agg({
        'value' : list,
        'model' : 'first',
        'head' : 'first',
        'score' : 'first',
        'k' : 'first',
    }).reset_index(drop = True)
    
    # Create the boxplots
    n_replicates: int = df['replicate'].nunique()
    width = 0.3  # width of the boxplot
    models: List[str] = df_grouped['model'].unique().tolist()
    shift_amt: int = 0
    for model in models:
        heads: List[str] = df_grouped[df_grouped['model'] == model]['head'].unique().tolist()
        for head in heads:
            df_ = df_grouped[(df_grouped['model'] == model) & (df_grouped['head'] == head)]
            full_data_values: np.ndarray = np.array([ [x] * n_replicates for x in df_[df_['k'] == full_data_k]['value'].tolist() ]).flatten() # expand to match # of replicates, since only use 1 replicate for `all`
            values: np.ndarray = np.array(df_[df_['k'] != full_data_k]['value'].tolist())

            # Get relative difference between full data v. few-shot
            values = values - full_data_values

            # create boxplots at positions shifted by the width + spacing btwn boxplots
            bp = ax.boxplot(values.tolist(), positions=positions + shift_amt * width - width / 2 + (shift_amt * 0.05), widths=width, showfliers=False, manage_ticks=False, patch_artist=True)
            shift_amt += 1

            # set the outline color
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                for box in bp[element]:
                    box.set(color=SCORE_MODEL_HEAD_2_COLOR[score][model][head])
            plt.setp(bp['boxes'], facecolor='white')
            # Increase the size of the median line and set to black
            for median_line in bp['medians']:
                median_line.set_linewidth(3)

    # Draw line at 0
    ax.axhline(0, color='black', linestyle='dashed')
    if task_group == 'chexpert':
        ax.text(0.2, 0.975, '▲ Few-shot better', transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, pad=0.3))
        ax.text(0.2, 0.05, '▼ Full data better', transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, pad=0.3))
    else:
        ax.text(0.2, 0.95, '▲ Few-shot better', transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, pad=0.3))
        ax.text(0.2, 0.07, '▼ Full data better', transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, pad=0.3))

    # show plot
    ax.set_xlabel("# of Train Examples per Class", fontsize=12)
    ax.set_ylabel(f'{score.upper()} gain from k-shot v. full data model', fontsize=12)
    ax.set_title(TASK_GROUP_2_PAPER_NAME[task_group], size=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticks(positions, labels=x_tick_labels)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_column_per_patient(df_demo: pd.DataFrame, 
                            path_to_output_dir: str,
                            column: str, 
                            x_label: str, 
                            title: str,
                            max_clamp: int = None):
    """
    3 panels, one for each split
        Histogram
            x-axis: # of events in a patient timeline
            y-axis: # of patients with that # of events
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for idx, split in enumerate(['train', 'val', 'test']):
        df_ = df_demo[df_demo['split'] == split]
        counts = df_[column].tolist()
        
        # Clamp at `max_clamp`
        if max_clamp:
            counts = [ min(count, max_clamp) for count in counts ]
        
        axes[idx].hist(counts, bins=100)
        axes[idx].set_xlabel(f"{x_label}")
        axes[idx].set_ylabel("# of Patients")
        axes[idx].legend()
        axes[idx].set_title(f'{split} (n={len(counts)})')
    fig.suptitle(f"Distribution of {title}/patient")
    plt.savefig(os.path.join(path_to_output_dir, f'{column}_per_patient.png'))
    plt.show()
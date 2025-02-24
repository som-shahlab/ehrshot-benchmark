import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    HEAD_2_INFO,
    MODEL_2_INFO, 
    TASK_GROUP_2_PAPER_NAME,
    SCORE_MODEL_HEAD_2_COLOR,
    filter_df,
)

def _plot_unified_legend(fig, axes, ncol=None, fontsize=8):
    """Create a unified legend for the entire figure."""
    labels = []
    label2handle = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                label2handle[l] = h
    legend_n_col: int = len([x for x in labels if '(Full)' not in x]) if ncol is None else ncol
    
    # Create grouped legends by model type
    model_types = {'CLIMBR': [], 'LLM': [], 'BERT': []}
    for label in labels:
        if 'clmbr' in label.lower():
            model_types['CLIMBR'].append(label)
        elif any(llm in label.lower() for llm in ['llm', 'gpt', 'qwen']):
            model_types['LLM'].append(label)
        else:
            model_types['BERT'].append(label)
    
    # Create legend with grouped models
    all_labels = []
    all_handles = []
    for model_type, type_labels in model_types.items():
        if type_labels:
            all_labels.extend(type_labels)
            all_handles.extend([label2handle[l] for l in type_labels])
    
    # Calculate optimal number of columns (max 3 for readability)
    ncols = min(len(all_handles), 4)
    
    # Let constrained_layout handle the legend positioning
    fig.legend(all_handles, all_labels, loc='center', 
               ncol=ncols, fontsize=fontsize, frameon=False,
               bbox_to_anchor=(0.5, -0.025))

def save_plot_data_to_feather(df: pd.DataFrame, path: str, filename: str):
    """Save plot data to a feather file, creating directories if they don't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    df.reset_index().to_feather(os.path.join(path, filename))

def plot_one_labeling_function(df: pd.DataFrame,
                              ax: plt.Axes,
                              labeling_function: str,
                              sub_tasks: List[str],
                              score: str,
                              path_to_output_dir: str,
                              model_heads: Optional[List[Tuple[str, str]]] = None,
                              is_x_scale_log: bool = True,
                              is_std_bars: bool = True):
    """
        Graph: Line plot of each model+head's results for a single labeling function as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            lines = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
    """
    # Limit to specific labeling_function, subtask, score, (model, head) combos
    df = filter_df(df, score=score, labeling_function=labeling_function, sub_tasks=sub_tasks, model_heads=model_heads)
    
    if df.shape[0] == 0:
        print(f"Skipping {labeling_function} because no results for {model_heads}")
        return

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
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            model_name: str = MODEL_2_INFO[model]['label']
            head_name: str = HEAD_2_INFO[head]['label']

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')
            df_stds_ = df_stds[(df_stds['model'] == model) & (df_stds['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual subtasks
            for subtask in df_means_['sub_task'].unique():
                df_m_ = df_means_[df_means_['sub_task'] == subtask]
                df_s_ = df_stds_[df_stds_['sub_task'] == subtask]
                # Plot individual subtasks with thinner lines
                ax.plot(df_m_['k'], df_m_['value'], color=color, linestyle='-', linewidth=1, alpha=0.15)
                if is_std_bars:
                    ax.fill_between(df_m_['k'], 
                                  df_m_['value'] - df_s_['value'], 
                                  df_m_['value'] + df_s_['value'],
                                  color=color, alpha=0.1)

            # Plot average line across all subtasks with improved styling
            df_ = df_means_.groupby(['k']).agg({'value': 'mean', 'k': 'first'}).reset_index(drop=True)
            marker = 'X' if 'clmbr' in model.lower() else ('o' if 'llm' in model.lower() else 'p')
            ax.plot(df_['k'], df_['value'], color=color, 
                   label=f'{model_name}+{head_name}',
                   linestyle='-', marker=marker, 
                   linewidth=1.5, markersize=8,
                   markeredgewidth=1.5,
                   markeredgecolor='white')

    # Enhanced plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
    else:
        ax.grid(True, ls="-", alpha=0.2)
        
    # Format axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major')
    
    # Set labels and title
    ax.set_title(LABELING_FUNCTION_2_PAPER_NAME[labeling_function], pad=10, fontweight='bold')
    ax.set_ylabel(score.upper(), fontweight='bold')
    ax.set_xlabel("# of Train Examples per Class", fontweight='bold')
    
    # Set ticks - only show powers of 2, "1", and "All"
    visible_ticks = []
    visible_labels = []
    for k, label in zip(ks, x_tick_labels):
        if label == "All" or label == "1" or (isinstance(k, (int, float)) and (k & (k - 1) == 0)):
            visible_ticks.append(k)
            visible_labels.append(label)
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels(visible_labels)
    
    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Save raw data to feather files
    raw_data_dir = os.path.join(path_to_output_dir, 'raw_data', 'labeling_functions')
    
    # Save the processed dataframes
    save_plot_data_to_feather(df_means, raw_data_dir, f'{labeling_function}_means.feather')
    save_plot_data_to_feather(df_stds, raw_data_dir, f'{labeling_function}_stds.feather')
    
    # Save the original filtered data
    save_plot_data_to_feather(df, raw_data_dir, f'{labeling_function}_raw.feather')


def plot_one_task_group(df: pd.DataFrame,
                        ax: plt.Axes, 
                        task_group: str, 
                        score: str,
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None, 
                        is_x_scale_log: bool = True):    
    """
        Graph: Aggregated line plot of each model+head's results for all of the labeling functions within a task group, as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            dark lines = model+head's achieved mean score across replicates, averaged across all labeling functions in this task group
            faded lines = model+head's achieved mean score across replicates for each individual labeling function
    """

    # Limit to specific task_group, score, (model, head) combos
    df = filter_df(df, score=score, task_group=task_group, model_heads=model_heads)

    if df.shape[0] == 0:
        print(f"Skipping {task_group} because no results for {model_heads}")
        return

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
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            model_name: str = MODEL_2_INFO[model]['label']
            head_name: str = HEAD_2_INFO[head]['label']

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual subtasks with thinner lines
            for subtask in df_means_['sub_task'].unique():
                df_m_ = df_means_[df_means_['sub_task'] == subtask]
                ax.plot(df_m_['k'], df_m_['value'], color=color, linestyle='-', linewidth=1, alpha=0.15)
    
            # Plot average line per model with improved styling
            df_ = df_means_.groupby(['k']).agg({'value': 'mean', 'k': 'first'}).reset_index(drop=True)
            marker = 'X' if 'clmbr' in model.lower() else ('o' if 'llm' in model.lower() else 'p')
            ax.plot(df_['k'], df_['value'], color=color, 
                   label=f'{model_name}+{head_name}',
                   linestyle='-', marker=marker, 
                   linewidth=1.5, markersize=8,
                   markeredgewidth=1.5,
                   markeredgecolor='white')

    # Enhanced plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
    else:
        ax.grid(True, ls="-", alpha=0.2)
        
    # Format axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Set labels and title
    ax.set_title(TASK_GROUP_2_PAPER_NAME[task_group], 
                 size=12, pad=10, fontweight='bold')
    ax.set_ylabel(score.upper(), fontsize=10, fontweight='bold')
    ax.set_xlabel("# of Train Examples per Class", fontsize=10, fontweight='bold')
    
    # Set ticks - only show powers of 2, "1", and "All"
    visible_ticks = []
    visible_labels = []
    for k, label in zip(ks, x_tick_labels):
        if label == "All" or label == "1" or (isinstance(k, (int, float)) and (k & (k - 1) == 0)):
            visible_ticks.append(k)
            visible_labels.append(label)
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels(visible_labels)
    
    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Save raw data to feather files
    raw_data_dir = os.path.join(path_to_output_dir, 'raw_data', 'task_groups')
    
    # Save the processed dataframes
    save_plot_data_to_feather(df_means, raw_data_dir, f'{task_group}_means.feather')
    
    # Save the original filtered data
    save_plot_data_to_feather(df, raw_data_dir, f'{task_group}_raw.feather')


def plot_one_task_group_box_plot(df: pd.DataFrame,
                                ax: plt.Axes,
                                task_group: str, 
                                score: str,
                                path_to_output_dir: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None):
    """
    Graph: Aggregated box plot containing each model+head's results for all of the labeling functions within a task group, as a function of `k`,
        where results are the relative difference between the `k`-shot model+head and the full data model+head.

        y-axis = model+head's achieved score across replicates and labeling functions within a task group, relative to its score with full data
        x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    """
    # Set style configuration
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
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

            # Set box colors and styles
            color = SCORE_MODEL_HEAD_2_COLOR[score][model][head]
            for element in ['boxes', 'whiskers', 'fliers', 'caps']:
                for box in bp[element]:
                    box.set(color=color)
            
            # White fill for boxes with colored edges
            plt.setp(bp['boxes'], facecolor='white', alpha=0.8)
            
            # Make median lines black and thicker for better visibility
            for median in bp['medians']:
                median.set(color='black', linewidth=2)
            
            # Make caps slightly thicker
            for cap in bp['caps']:
                cap.set(linewidth=1.5)

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

    # Enhanced plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    
    # Format axes and labels
    ax.set_xlabel("# of Train Examples per Class", fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{score.upper()} gain from k-shot v. full data model', fontsize=10, fontweight='bold')
    ax.set_title(TASK_GROUP_2_PAPER_NAME[task_group], size=12, pad=10, fontweight='bold')
    
    # Adjust ticks - only show powers of 2, "1", and "All"
    visible_positions = []
    visible_labels = []
    for pos, label in zip(positions, x_tick_labels):
        if str(label) == "1" or (isinstance(label, (int, float)) and (label & (label - 1) == 0)):
            visible_positions.append(pos)
            visible_labels.append(str(label))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(visible_positions, labels=visible_labels)
    
    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Save raw data to feather files
    raw_data_dir = os.path.join(path_to_output_dir, 'raw_data', 'task_groups_boxplots')
    
    # Save the processed dataframes
    save_plot_data_to_feather(df_grouped, raw_data_dir, f'{task_group}_grouped.feather')
    
    # Save the original filtered data
    save_plot_data_to_feather(df, raw_data_dir, f'{task_group}_raw.feather')

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
    # Set style configuration
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Create figure with constrained layout
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 5), constrained_layout=True)
    
    for idx, split in enumerate(['train', 'val', 'test']):
        df_ = df_demo[df_demo['split'] == split]
        counts = df_[column].tolist()
        
        # Clamp at `max_clamp`
        if max_clamp:
            counts = [min(count, max_clamp) for count in counts]
        
        # Enhanced histogram
        axes[idx].hist(counts, bins=50, color='#1f77b4', alpha=0.75, edgecolor='white')
        
        # Format axes
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].grid(True, ls='-', alpha=0.2)
        
        # Labels
        axes[idx].set_xlabel(f"{x_label}", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("# of Patients", fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{split.capitalize()} (n={len(counts)})', 
                           size=12, pad=10, fontweight='bold')
        
        # Adjust ticks
        axes[idx].tick_params(axis='both', which='major', labelsize=8)
    
    # Main title
    fig.suptitle(f"Distribution of {title}/patient", 
                 fontsize=14, fontweight='bold')
    
    # Save with high quality
    plt.savefig(os.path.join(path_to_output_dir, f'{column}_per_patient.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

import os
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Patch
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    HEAD_2_INFO,
    MODEL_2_INFO, 
    SHOT_STRATS,
    SCORE_MODEL_HEAD_2_COLOR,
    filter_df,
    type_tuple_list,
)
from plot import (
    plot_one_labeling_function,
    plot_one_task_group,
    plot_one_task_group_box_plot,
    _plot_unified_legend,
)

def plot_all_labeling_functions(df_results: pd.DataFrame, 
                              score: str, 
                              path_to_output_dir: str,
                              model_heads: Optional[List[Tuple[str, str]]] = None,
                              is_x_scale_log: bool = True,
                              is_std_bars: bool = True):
    # Set style configuration
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Create figure with A4 size and constrained layout
    fig, axes = plt.subplots(5, 3, figsize=(8.27, 11.69), constrained_layout=True)  # A4 size (210mm × 297mm)
    labeling_functions: List[str] = df_results[df_results['score'] == score]['labeling_function'].unique().tolist()
    for idx, labeling_function in enumerate(labeling_functions):
        sub_tasks: List[str] = df_results[(df_results['score'] == score) & (df_results['labeling_function'] == labeling_function)]['sub_task'].unique().tolist()
        plot_one_labeling_function(df_results, 
                                    axes.flat[idx], 
                                    labeling_function, 
                                    sub_tasks, 
                                    score,
                                    path_to_output_dir=path_to_output_dir,
                                    model_heads=model_heads,
                                    is_x_scale_log=is_x_scale_log,
                                    is_std_bars=False if labeling_function == 'chexpert' else is_std_bars)

    # Enhanced plot aesthetics
    fig.suptitle(f'{score.upper()} by Task', fontsize=12, fontweight='bold')
    
    _plot_unified_legend(fig, axes, fontsize=8)
    
    # Save as both PNG and PDF
    plt.savefig(os.path.join(path_to_output_dir, f"tasks_{score}.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(path_to_output_dir, f"tasks_{score}.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig

def plot_all_task_groups(df_results: pd.DataFrame, 
                        score: str, 
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None,
                        is_x_scale_log: bool = True):
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
    
    # Create figure with A4 size and constrained layout
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 6), constrained_layout=True)  # A4 size (210mm × )
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in enumerate(task_groups):
        plot_one_task_group(df_results, 
                            axes.flat[idx], 
                            task_group, 
                            score,
                            path_to_output_dir=path_to_output_dir,
                            model_heads=model_heads,
                            is_x_scale_log=is_x_scale_log)
    
    # Enhanced plot aesthetics
    fig.suptitle(f'{score.upper()} by Task Group', 
                 fontsize=16, fontweight='bold')
    
    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, fontsize=8)
    
    # Save as both PNG and PDF
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_{score}.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_{score}.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig

def plot_all_task_group_box_plots(df_results: pd.DataFrame,
                                 score: str, 
                                 path_to_output_dir: str,
                                 model_heads: Optional[List[Tuple[str, str]]] = None):
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
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Create figure with A4 size and constrained layout
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69), constrained_layout=True)  # A4 size (210mm × 297mm)
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in tqdm(enumerate(task_groups)):
        plot_one_task_group_box_plot(df_results, 
                                    axes.flat[idx], 
                                    task_group, 
                                    score,
                                    path_to_output_dir=path_to_output_dir,
                                    model_heads=model_heads)
    
    # Create a unified legend grouped by model type
    df_ = filter_df(df_results, score=score, model_heads=model_heads)
    model_types = {'CLIMBR': [], 'LLM': [], 'BERT': []}
    
    # Group models by type
    for model, head in df_[['model', 'head']].drop_duplicates().itertuples(index=False):
        label = f"{MODEL_2_INFO[model]['label']}+{HEAD_2_INFO[head]['label']}"
        patch = Patch(
            facecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head],
            edgecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head],
            label=label
        )
        if 'clmbr' in model.lower():
            model_types['CLIMBR'].append(patch)
        elif any(llm in model.lower() for llm in ['llm', 'gpt', 'qwen']):
            model_types['LLM'].append(patch)
        else:
            model_types['BERT'].append(patch)
    
    # Combine handles in grouped order
    handles = []
    for model_type, patches in model_types.items():
        if patches:
            handles.extend(patches)
    
    # Create legend with grouped models in a single row
    fig.legend(handles=handles, loc='center', 
               ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    
    # Enhanced plot aesthetics
    fig.suptitle(f'Few-shot v. Full data {score.upper()} by Task Group', 
                 fontsize=16, fontweight='bold')
    # Save as both PNG and PDF
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_boxplot_{score}.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_boxplot_{score}.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig


def plot_radar_chart(df_results: pd.DataFrame, k: int, path_to_output_dir: str, score: str = "auroc", model_heads: Optional[List[Tuple[str, str]]] = None):
    """
    Create a radar (spider) plot comparing model+head performance across labeling functions for a given k.
    
    The function filters the results for a given score (e.g. "auroc") and training set size k,
    aggregates over subtasks (taking the mean), and creates a radar plot using plotly.
    
    Args:
        df_results: DataFrame containing evaluation results.
        k: The number of training examples per class to plot.
        path_to_output_dir: Directory where the radar plot PNG will be saved.
        score: The performance score to filter on (default "auroc").
        model_heads: Optional list of (model, head) tuples to restrict plotting.
    
    Returns:
        The plotly Figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Filter data for desired score, k, and optional model_heads
    df = filter_df(df_results, score=score, model_heads=model_heads)
    if k == -1 and (-1 in df['k'].unique()):
        # Handle 'full data' case if -1 is in the dataset
        ks_nonneg = [x for x in df['k'].unique() if x != -1]
        k_plot = 2 * max(ks_nonneg) if ks_nonneg else -1
        df.loc[df['k'] == -1, 'k'] = k_plot
    else:
        k_plot = k
    df_k = df[df['k'] == k_plot]
    if df_k.empty:
        print(f"No data for k={k_plot} and score={score}.")
        return None

    # Aggregate over subtasks for each labeling_function, model, and head
    df_grouped = df_k.groupby(['labeling_function', 'model', 'head']).agg({'value': 'mean'}).reset_index()

    # Remove Llama models
    df_grouped = df_grouped[~df_grouped['model'].str.lower().str.contains('llama')]
    if df_grouped.empty:
        print("No data available after filtering out Llama.")
        return None

    # Create model+head labels with custom mapping
    def get_custom_model_name(model, head):
        model_head = f"{MODEL_2_INFO[model]['label']}+{HEAD_2_INFO[head]['label']}"
        if model_head == "CLMBR+LR":
            return "EHR Foundation Model"
        elif model_head == "Count-based+GBM":
            return "Counts Baseline"
        elif model_head == "GTE Qwen2 7B+LR":
            return "LLM Encoder"
        return model_head
    
    df_grouped['model_head'] = df_grouped.apply(lambda x: get_custom_model_name(x['model'], x['head']), axis=1)

    # Export data for R visualization
    export_df = df_grouped[['model_head', 'labeling_function', 'value']].copy()
    export_path = os.path.join(path_to_output_dir, f"radar_data_{score}_k{k}.csv")
    export_df.to_csv(export_path, index=False)

    # Rename labeling functions to paper names
    df_grouped['category'] = df_grouped['labeling_function'].map(lambda x: LABELING_FUNCTION_2_PAPER_NAME.get(x, x))

    # Create figure with white background
    fig = go.Figure()

    # Define the desired order of model+head combinations
    desired_order = ["Counts Baseline", "EHR Foundation Model", "LLM Encoder"]
    
    # Add traces in the specified order
    for model_name in desired_order:
        df_model = df_grouped[df_grouped['model_head'] == model_name]
        if df_model.empty:
            continue
            
        model = df_model['model'].iloc[0]
        head = df_model['head'].iloc[0]
            
        # Get color from the same color scheme as matplotlib plots
        color = SCORE_MODEL_HEAD_2_COLOR[score][model][head]
        
        # Handle color conversion safely
        try:
            # Convert matplotlib color names to RGB values
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
            
            # Convert any matplotlib color name to RGB
            rgb = mcolors.to_rgba(color)[:3]
            
            fill_color = f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},0.2)'
            line_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
            
            # Add filled area trace
            fig.add_trace(go.Scatterpolar(
                r=df_model['value'].tolist() + [df_model['value'].iloc[0]],  # Close the polygon
                theta=df_model['category'].tolist() + [df_model['category'].iloc[0]],  # Close the polygon
                name=df_model['model_head'].iloc[0],
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=line_color, width=3),  # Increased from 2
                mode='lines+markers',
                # Use the same markers as matplotlib plots
                marker=dict(
                    size=8,
                    symbol='x' if 'clmbr' in model.lower() else ('circle' if 'llm' in model.lower() else 'star'),
                    line=dict(color='white', width=1)
                )
            ))
        except Exception as e:
            print(f"Warning: Color conversion failed for {color}, using default color")
            # Use a default color if conversion fails
            fig.add_trace(go.Scatterpolar(
                r=df_model['value'].tolist() + [df_model['value'].iloc[0]],
                theta=df_model['category'].tolist() + [df_model['category'].iloc[0]],
                name=df_model['model_head'].iloc[0],
                fill='toself',
                fillcolor='rgba(128,128,128,0.2)',  # Default gray with transparency
                line=dict(color='rgb(128,128,128)', width=4),
                mode='lines+markers',
                marker=dict(size=8, symbol='circle', line=dict(color='white', width=1))
            ))

    # Update layout for consistent style with matplotlib
    fig.update_layout(
        title=dict(
            text=f"{score.upper()} in Few-Shot Setting",
            font=dict(size=20, family='Arial', weight='bold'),  # Increased from 16
            y=0.99
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=16, family='Arial'),  # Increased from 14
            orientation='h',
            yanchor='bottom',
            y=-0.15,  # Increased spacing from -0.05
            xanchor='center',
            x=0.5
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.4, 0.9],
                tickfont=dict(size=14, family='Arial', color='rgba(0,0,0,0.1)'),  # Increased from 12
                gridcolor='rgba(1,1,1,0.1)',
                tickcolor='rgba(0,0,0,0.1)',
                linecolor='rgba(0,0,0,0.1)',
                linewidth=2  # Added line thickness
            ),
            angularaxis=dict(
                tickfont=dict(size=16, family='Arial'),  # Increased from 14
                linecolor='rgba(0,0,0,0.5)',
                gridcolor='rgba(0,0,0,0.1)',
                linewidth=3  # Increased from 2
            ),
            bgcolor='white'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=70, b=50, l=170, r=170, autoexpand=True)  # Increased margins
    )

    # Save as PNG and PDF with high DPI
    outfile_png = os.path.join(path_to_output_dir, f"radar_{score}_k{k}.png")
    outfile_pdf = os.path.join(path_to_output_dir, f"radar_{score}_k{k}.pdf")
    fig.write_image(outfile_png, scale=2, width=900, height=750)
    fig.write_image(outfile_pdf, scale=2, width=900, height=750)
    
    # Also save interactive HTML version
    outfile_html = os.path.join(path_to_output_dir, f"radar_{score}_k{k}.html")
    fig.write_html(outfile_html)
    
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
    parser.add_argument("--model_heads", type=type_tuple_list, default=[], help="Specific (model, head) combinations to plot. Format it as a Python list of tuples of strings, e.g. [('clmbr', 'lr'), ('count', 'gbm')]")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_RESULTS_DIR: str = args.path_to_results_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    SHOT_STRAT: str = args.shot_strat
    MODEL_HEADS: Optional[List[Tuple[str, str]]] = args.model_heads if len(args.model_heads) > 0 else None
    # Create main output directory and raw data subdirectories
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'raw_data', 'labeling_functions'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'raw_data', 'task_groups'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'raw_data', 'task_groups_boxplots'), exist_ok=True)
    
    # Load all results from CSVs
    dfs: List[pd.DataFrame] = []
    for idx, labeling_function in tqdm(enumerate(LABELING_FUNCTION_2_PAPER_NAME.keys())):
        path_to_csv = os.path.join(PATH_TO_RESULTS_DIR, f"{labeling_function}/{SHOT_STRAT}_results.csv")
        if not os.path.exists(path_to_csv): 
            print("Skipping ", labeling_function)
            continue
        dfs.append(pd.read_csv(path_to_csv))
    df_results: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        
    
    ####################################
    ####################################
    #
    # Tables
    #
    ####################################
    ####################################
    
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
    
    # Table for each (labeling function, score)
    #   Rows = model + head
    #   Columns = k
    #   Cells = mean ± std of score
    for score in df_means['score'].unique():
        path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'individual_tasks', score)
        for sub_task in df_means['sub_task'].unique():
            os.makedirs(path_to_output_dir_, exist_ok=True)
            df_ = filter_df(df_means, sub_tasks=[sub_task], score=score, model_heads=MODEL_HEADS).sort_values(by=['model', 'head', 'k'])
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
            df_ = df_.rename(columns={'-1' : 'All'})
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
    for score in df_means['score'].unique():
        path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'task_groups', score)
        for task_group in task_groups:
            os.makedirs(path_to_output_dir_, exist_ok=True)
            df_ = filter_df(df_means, task_group=task_group, score=score, model_heads=MODEL_HEADS)
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

    print("Plotting Models: ", MODEL_HEADS)

    # Plotting individual AUROC/AUPRC plot for each labeling function
    for score in tqdm(df_results['score'].unique(), desc='plot_all_labeling_functions()'):
        if score == 'brier': continue
        plot_all_labeling_functions(df_results, score, PATH_TO_OUTPUT_DIR, 
                                    model_heads=MODEL_HEADS, is_x_scale_log=True, is_std_bars=True)

    # Plotting aggregated auroc and auprc plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_groups()'):
        if score == 'brier': continue
        plot_all_task_groups(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR, 
                             model_heads=MODEL_HEADS, is_x_scale_log=True)

    # plotting aggregated auroc and auprc box plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_group_box_plots()'):
        if score == 'brier': continue
        plot_all_task_group_box_plots(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR,
                                      model_heads=MODEL_HEADS)
                                      
    # Generate radar plots for different k values and scores
    k_values = [8, 16, 32, 128] 
    for score in tqdm(df_results['score'].unique(), desc='plot_radar_charts()'):
        if score == 'brier': continue
        for k in k_values:
            plot_radar_chart(df_results, k, PATH_TO_OUTPUT_DIR, score=score, model_heads=MODEL_HEADS)

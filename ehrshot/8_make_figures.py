import json
import os
import argparse
from utils import load_data
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.patches import Patch
from utils import LABELING_FUNCTIONS, LABELING_FUNCTION_2_PAPER_NAME

task_group_dict = {
    "operational_outcomes": [
        "guo_los",
        "guo_readmission",
        "guo_icu"
    ],
    "lab_values": [
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ],
    "new_diagnoses": [
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi"
    ],
    "chexpert": [
        "chexpert"
    ]
}

def get_avg_data(data_dict, score="auroc"):
    scores = []
    for i in data_dict:
        scores.append(data_dict[i]["scores"][score])
    
    scores = np.array(scores)
    avg_score = np.mean(scores, axis=0)
    return avg_score

def plot_individual_results(path_to_eval: dict, 
                 labeling_function: str, 
                 size: int = 14,
                 path_to_save: str = "./"):
    """label_dict[labeling_function][model_name][replicate][scores][auroc]"""

    label_dict = json.load(os.path.join(path_to_eval, "few_tune_params_True.json"), 'r')
    label_dict_long = json.load(os.path.join(path_to_eval, "long_tune_params_True.json"), 'r')

    task = label_dict[labeling_function]
    task_long = label_dict_long[labeling_function]

    models = list(task.keys())
    scores = list(task[models[0]]["0"]["scores"].keys()) # e.g. auroc, ap, mse
    n_replicates: int = len(task[models[0]])
    for s in sorted(scores, reverse=True):
        if s == "auroc":
            colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        elif s == "auprc":
            colors = ['#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        else:
            continue
        fig, ax = plt.subplots()
        plt.xscale("log")
        for idx, m in enumerate(models):
            if m == "Codes_Only":
                legend_name = "CLMBR"
                legend_name_long = "CLMBR Full"
            elif m == "Count_based_GBM":
                legend_name = "GBM"
                legend_name_long = "GBM Full"
            all_values = [] # Collect values across all replicates
            all_values_long = []
            for replicate in task[m].keys():
                all_values.append(task[m][replicate]["scores"][s])
                if replicate == "0":
                    all_values_long.append(task_long[m][replicate]["scores"][s])
            all_values = np.array(all_values)
            means = np.mean(all_values, axis=0)
            means_long = np.mean(all_values_long)
            stds = np.std(all_values, axis=0)
            x = task[m]["0"]['k']
            color = colors[idx]

            plt.plot(x, means, color=color, label=legend_name, linestyle='-', marker='o', linewidth=3, markersize=7)
            plt.plot(x, means - 0.5 * stds, color=color, alpha=0.1)
            plt.plot(x, means + 0.5 * stds, color=color, alpha=0.1)
            plt.fill_between(x, means - 0.5 * stds, means + 0.5 * stds, color=color, alpha=0.2, )
            plt.axhline(y=means_long, color=color, linestyle='--', linewidth=3, label=legend_name_long)
        
        plt.legend(fontsize=12)
        plt.title(labeling_function, size=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xticks(x, x)
        plt.tight_layout()
        new_path_to_save = os.path.join(path_to_save, f"{labeling_function}_{s}.png")
        plt.savefig(new_path_to_save, dpi=300)
        plt.close('all')


def make_data_dict(path_to_eval, task_group):
    gbm_aurocs = []
    clmbr_aurocs = []
    gbm_auprcs = []
    clmbr_auprcs = []

    LABELING_FUNCTIONS = task_group_dict[task_group]

    for labeling_function in tqdm(LABELING_FUNCTIONS):
        data = json.load(os.path.join(path_to_eval, f"{labeling_function}/few_tune_params_True.json"), 'r')

        for label_name in data:
            gbm_dict = data[label_name]["Count_based_GBM"]
            clmbr_dict = data[label_name]["Codes_Only"]

            gbm_auroc = get_avg_data(gbm_dict, "auroc")
            gbm_auprc = get_avg_data(gbm_dict, "auprc")
            clmbr_auroc = get_avg_data(clmbr_dict, "auroc")
            clmbr_auprc = get_avg_data(clmbr_dict, "auprc")

            gbm_aurocs.append(gbm_auroc)
            gbm_auprcs.append(gbm_auprc)
            clmbr_aurocs.append(clmbr_auroc)
            clmbr_auprcs.append(clmbr_auprc)


    k = gbm_dict["0"]["k"]

    gbm_aurocs_long = []
    clmbr_aurocs_long = []

    gbm_auprcs_long = []
    clmbr_auprcs_long = []

    for labeling_function in tqdm(LABELING_FUNCTIONS):
        data = json.load(os.path.join(path_to_eval, f"{labeling_function}/long_tune_params_True.json"), 'r')

        for label_name in data:
            gbm_dict = data[label_name]["Count_based_GBM"]
            clmbr_dict = data[label_name]["Codes_Only"]

            gbm_auroc = get_avg_data(gbm_dict, "auroc")
            gbm_auprc = get_avg_data(gbm_dict, "auprc")
            clmbr_auroc = get_avg_data(clmbr_dict, "auroc")
            clmbr_auprc = get_avg_data(clmbr_dict, "auprc")

            gbm_aurocs_long.append(gbm_auroc)
            gbm_auprcs_long.append(gbm_auprc)
            clmbr_aurocs_long.append(clmbr_auroc)
            clmbr_auprcs_long.append(clmbr_auprc)

    auroc_dict = {
        "GBM": gbm_aurocs, 
        "CLMBR": clmbr_aurocs, 
        "GBM_LONG": gbm_aurocs_long, 
        "CLMBR_LONG": clmbr_aurocs_long
    }

    auprc_dict = {
        "GBM": gbm_auprcs, 
        "CLMBR": clmbr_auprcs, 
        "GBM_LONG": gbm_auprcs_long, 
        "CLMBR_LONG": clmbr_auprcs_long
    }

    data_dict = {
        "auroc": auroc_dict, 
        "auprc": auprc_dict
    }

    return data_dict, k


def plot_agg_results(
                data_dict, task_group,
                k, 
                size = 14,
                save_path="./"):

    for score in data_dict:
        if score == "auroc":
            colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
            # colors = ["green", "red", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        elif score == "auprc":
            # colors = ["blue", "orange", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            colors = ['#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        else:
            continue
    
        score_dict = data_dict[score]
        plt.subplots()
        plt.xscale("log")
        for idx, model in enumerate(score_dict):
            if model not in ["GBM", "CLMBR"]:
                continue
            data = score_dict[model]
            color = colors[idx]
            for arr in data:
                plt.plot(k, arr, color=color, linestyle='-', linewidth=2, alpha=0.25)
        
            data_mean = np.mean(data, axis=0)
            # gbm_long = np.mean(data, axis=0)
            plt.plot(k, data_mean, color=color, label=model, linestyle='-', linewidth=3, marker='o', markersize=7)

            long_mean = np.mean(score_dict[f"{model}_LONG"])
            plt.axhline(y=long_mean, color=color, linestyle='--', linewidth=3, label=f'{model} Full')

            plt.xticks(k, k)

        plt.legend(fontsize=12)
        # plt.title("Chest X-ray Findings")
        plt.xlabel("# of Train Examples per Class", fontsize=size)
        plt.ylabel(score.upper(), fontsize=size)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        path_to_save = os.path.join(save_path, f"{task_group}_{score}_agg.png")
        plt.savefig(path_to_save, dpi=300)
        plt.close('all')


def plot_fewshot_over_full_boxplot(data_dict, task_group,
                 metric: str,
                 size = 14,
                 save_path="./"):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get data
    xs = list(data_dict['clmbr'].keys())
    positions = np.arange(len(xs))
    
    # Create the boxplots
    width = 0.3  # width of the boxplot
    if metric == 'auroc':
        COLORS = { 'clmbr' : '#ff7f00', 'gbm' : '#377eb8' }
    elif metric == 'auprc':
        COLORS = { 'clmbr' : '#f781bf', 'gbm' : '#4daf4a' }

    for i, (model, k_shot_dict) in enumerate(data_dict.items()):
        ys: list[float] = list(k_shot_dict.values())
        # create boxplots at positions shifted by the width + spacing btwn boxplots
        bp = ax.boxplot(ys, positions=positions + i * width - width / 2 + (i * 0.05), widths=width, showfliers=False, manage_ticks=False, patch_artist=True)
        
        # set the outline color
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for box in bp[element]:
                box.set(color=COLORS[model])
        plt.setp(bp['boxes'], facecolor='white')
        # Increase the size of the median line and set to black
        for median_line in bp['medians']:
            median_line.set_linewidth(3)

        # add points
        # for (x,y) in zip(positions, ys):
        #     # Add some random "jitter" to the x-axis
        #     x = np.random.normal(x+ i * width - width / 2, 0.04, size=len(y))
        #     plt.scatter(x, y, color='red', s=8)

    # legend
    legend_elements = [ Patch(facecolor=color, edgecolor=color, label=model.upper()) for model, color in COLORS.items() ]
    ax.legend(handles=legend_elements)
    
    # Draw line at 0
    plt.axhline(0, color='black', linestyle='dashed')
    if task_group == 'chexpert':
        plt.text(0.6, 0.975, '▲ Few-shot better', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, pad=0.5))
        plt.text(0.6, 0.05, '▼ Full data better', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, pad=0.4))
    else:
        plt.text(0.4, 0.95, '▲ Few-shot better', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, pad=0.5))
        plt.text(0.4, 0.07, '▼ Full data better', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, pad=0.4))

    # show plot
    plt.xlabel("# of Train Examples per Class", fontsize=size)
    plt.ylabel(f'{metric.upper()} gain from using few-shot over full data model', fontsize=size)
    plt.yticks(fontsize=10)
    ax.set_xticks(positions, labels=xs, fontsize=10)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()
    path_to_save = os.path.join(save_path, f"{task_group}_{metric}_full_comparison_boxplot.png")
    plt.savefig(path_to_save, dpi=300)
    plt.close('all')

def plot_clmbr_over_gbm_boxplot(data_dict, task_group,
                 metric: str,
                 size = 14,
                 save_path="./"):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get data
    xs = list(data_dict.keys())
    positions = range(len(xs))
    ys = [ data_dict[x] for x in xs ]
    
    # Create the boxplot
    bp = ax.boxplot(ys, positions=positions, vert=True, showfliers=False)
    
    # Increase the size of the median line and set to black
    for median_line in bp['medians']:
        median_line.set_linewidth(3)
        median_line.set_color('black')

    # add points
    for (x,y) in zip(positions, ys):
        # Add some random "jitter" to the x-axis
        x = np.random.normal(x, 0.04, size=len(y))
        plt.scatter(x, y, color='blue', s=8)

    # x-axis labels
    ax.set_xticklabels(xs)
    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    # Draw line at 0
    plt.axhline(0, color='black', linestyle='dashed')
    plt.text(0.8, 0.95, '▲ CLMBR better', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, pad=0.5))
    plt.text(0.8, 0.07, '▼ CLMBR worse', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, pad=0.5))

    # show plot
    plt.xlabel("# of Train Examples per Class", fontsize=size)
    plt.ylabel(f'{metric.upper()} gain with CLMBR over GBM', fontsize=size)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    path_to_save = os.path.join(save_path, f"{task_group}_{metric}_boxplot.png")
    plt.savefig(path_to_save, dpi=300)
    plt.close('all')


def make_box_plot(path_to_eval, 
                  task_group, 
                  PLOT_DIFF_BETWEEN_MODELS: bool = False, 
                  PLOT_DIFF_BETWEEN_FULL: bool = True, 
                  save_path="./"):
    METRICS = ['auroc', 'auprc']
    MODELS = ['Count_based_GBM', 'Codes_Only', ]
    SHOTS = []

    LABELING_FUNCTIONS = task_group_dict[task_group]

    fewshot_results = {
        'Count_based_GBM' : {
            m: {} # [key] = labeling_function, [value] = dict where [key] = k, [value] = list of scores across replicates
            for m in METRICS
        },
        'Codes_Only' : {
            m: {} # [key] = labeling_function, [value] = dict where [key] = k, [value] = list of scores across replicates
            for m in METRICS
        },
    }
    fulldata_results = {
        'Count_based_GBM' : {
            m: {} # [key] = labeling_function, [value] = dict where [key] = k, [value] = list of scores across replicates
            for m in METRICS
        },
        'Codes_Only' : {
            m: {} # [key] = labeling_function, [value] = dict where [key] = k, [value] = list of scores across replicates
            for m in METRICS
        },
    }

    for labeling_function in tqdm(LABELING_FUNCTIONS):
        # Load few-shot data
        data = json.load(os.path.join(path_to_eval, f"{labeling_function}/few_tune_params_True.json"), 'r')
        for model in MODELS:
            for metric in METRICS:
                for label_str in data:
                    fewshot_results[model][metric][labeling_function] = collections.defaultdict(list) # [key] = k, [value] = list of scores across replicates
                    for replicate, scores_dict in data[label_str][model].items():
                        SHOTS = scores_dict['k']
                        for k_idx, k in enumerate(scores_dict['k']):
                            fewshot_results[model][metric][labeling_function][k].append(scores_dict['scores'][metric][k_idx])
        # Load full data
        long_data = json.load(os.path.join(path_to_eval, f"{labeling_function}/long_tune_params_True.json"), 'r')
        for model in MODELS:
            for metric in METRICS:
                for label_str in long_data:
                    fulldata_results[model][metric][labeling_function] = collections.defaultdict(list) # [key] = k, [value] = list of scores across replicates
                    for replicate, scores_dict in long_data[label_str][model].items():
                        for k_idx, k in enumerate(scores_dict['k']):
                            fulldata_results[model][metric][labeling_function][k].append(scores_dict['scores'][metric][k_idx])

    # Save each model separately for easy reference
    clmbr = fewshot_results['Codes_Only']
    gbm = fewshot_results['Count_based_GBM']
    full_clmbr = fulldata_results['Codes_Only']
    full_gbm = fulldata_results['Count_based_GBM']
    full_clmbr = {
        metric: {
            k: np.mean(np.concatenate([ full_clmbr[metric][labeling_function][k] for labeling_function in full_clmbr[metric].keys() ]))
            for k in full_clmbr[metric][list(full_clmbr[metric].keys())[0]].keys()
        }
        for metric in full_clmbr.keys()
    }
    full_gbm = {
        metric: {
            k: np.mean(np.concatenate([ full_gbm[metric][labeling_function][k] for labeling_function in full_gbm[metric].keys() ]))
            for k in full_gbm[metric][list(full_gbm[metric].keys())[0]].keys()
        }
        for metric in full_gbm.keys()
    }

    PATH_TO_SAVE: str = os.path.join(save_path)

    # Calculate differences between models at each K
    if PLOT_DIFF_BETWEEN_MODELS:
        y_values = collections.defaultdict(list) # [key] = k, [values] = delta in AUROC between each model replicate
        for metric in METRICS:
            for k in SHOTS:
                for labeling_function in tqdm(LABELING_FUNCTIONS):
                    assert set(gbm[metric][labeling_function].keys()) == set(clmbr[metric][labeling_function].keys()), f"Diff k shots between models"
                    clmbr_scores, gbm_scores = [], []
                    for replicate in range(len(gbm[metric][labeling_function][k])):
                        clmbr_scores.append(clmbr[metric][labeling_function][k][replicate])
                        gbm_scores.append(gbm[metric][labeling_function][k][replicate])
                    clmbr_scores = sorted(clmbr_scores)
                    gbm_scores = sorted(gbm_scores)
                    for (c, g) in zip(clmbr_scores, gbm_scores):
                        diff: float = c - g
                        y_values[k].append(diff)
            plot_clmbr_over_gbm_boxplot(y_values, metric=metric, task_group=task_group, save_path=PATH_TO_SAVE)

    # Calculate differences between each model v. full data performance at each K
    if PLOT_DIFF_BETWEEN_FULL:
        clmbr_y_values = collections.defaultdict(list) # [key] = k, [values] = delta in AUROC between each replicate and full data version
        gbm_y_values = collections.defaultdict(list) # [key] = k, [values] = delta in AUROC between each replicate and full data version
        for metric in METRICS:
            for k in SHOTS:
                for labeling_function in tqdm(LABELING_FUNCTIONS):
                    assert set(gbm[metric][labeling_function].keys()) == set(clmbr[metric][labeling_function].keys()), f"Diff k shots between models"
                    clmbr_scores, gbm_scores = [], []
                    for replicate in range(len(gbm[metric][labeling_function][k])):
                        clmbr_scores.append(clmbr[metric][labeling_function][k][replicate])
                        gbm_scores.append(gbm[metric][labeling_function][k][replicate])
                    clmbr_scores = sorted(clmbr_scores)
                    gbm_scores = sorted(gbm_scores)
                    for c in clmbr_scores:
                        diff: float = c - full_clmbr[metric][-1]
                        clmbr_y_values[k].append(diff)
                    for g in gbm_scores:
                        diff: float = g - full_gbm[metric][-1]
                        gbm_y_values[k].append(diff)

            plot_fewshot_over_full_boxplot({ 'clmbr' : clmbr_y_values, 'gbm' : gbm_y_values },  metric=metric, task_group=task_group, save_path=PATH_TO_SAVE)


def plot_results(label_dict: dict, 
                 labeling_function: str, 
                 size: int = 14,
                 path_to_save: str = "./"):
    """label_dict[labeling_function][model_name][replicate][scores][auroc]"""
    task = label_dict[labeling_function]
    models = list(task.keys())
    scores = list(task[models[0]][0]["scores"].keys()) # e.g. auroc, ap, mse
    n_replicates: int = len(task[models[0]])
    for s in sorted(scores, reverse=True):
        if s == "auroc":
            colors = ["green", "red", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        elif s == "auprc":
            colors = ["blue", "orange", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        else:
            continue
        fig, ax = plt.subplots()
        for idx, m in enumerate(models):
            if m == "Codes_Only":
                legend_name = "CLMBR"
            elif m == "Count_based_GBM":
                legend_name = "GBM"
            all_values = [] # Collect values across all replicates
            for replicate in task[m].keys():
                all_values.append(task[m][replicate]["scores"][s])
            all_values = np.array(all_values)
            means = np.mean(all_values, axis=0)
            stds = np.std(all_values, axis=0)
            x = task[m][0]['k']
            x = [x_i*4 for x_i in x]
            color = colors[idx]
            plt.plot(x, means, color=color, label=legend_name, linestyle='--', marker='o')
            plt.plot(x, means - 0.5 * stds, color=color, alpha=0.1)
            plt.plot(x, means + 0.5 * stds, color=color, alpha=0.1)
            plt.fill_between(x, means - 0.5 * stds, means + 0.5 * stds, color=color, alpha=0.2)
        
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # plt.legend(fontsize=size)
        # plt.xlabel("# of Train Examples per Class", fontsize=size)
        # plt.ylabel(s.upper(), fontsize=size)
        plt.title(LABELING_FUNCTION_2_PAPER_NAME[labeling_function], size=8)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)
        plt.tight_layout()
        new_path_to_save = os.path.join(path_to_save, f"{labeling_function}_{s}.png")
        plt.savefig(new_path_to_save, dpi=300)
        plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(description="Make plots of results")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to directory containing saved labels and featurizers")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory to save figures")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Plotting individual auroc and auprc plots
    for labeling_function in tqdm(LABELING_FUNCTIONS):
        label_dict = json.load(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, f"{labeling_function}/few_tune_params_True.json"), 'r')
        for lf in label_dict:
            path_to_labeling_function_eval = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, labeling_function)
            plot_individual_results(path_to_labeling_function_eval, labeling_function=lf, path_to_save=PATH_TO_OUTPUT_DIR)

    # Plotting aggregated auroc and auprc plots by task groups
    for task_group in tqdm(task_group_dict):
        data_dict, k = make_data_dict(PATH_TO_LABELS_AND_FEATS_DIR, task_group)
        plot_agg_results(data_dict, k=k, save_path=PATH_TO_OUTPUT_DIR, task_group=task_group)
    
    # plotting aggregated auroc and auprc box plots by task groups
    for task_group in tqdm(task_group_dict):
        make_box_plot(PATH_TO_LABELS_AND_FEATS_DIR, task_group=task_group, save_path=PATH_TO_OUTPUT_DIR)

    for labeling_function in tqdm(LABELING_FUNCTIONS):
        label_dict = json.load(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, f"{labeling_function}/few_tune_params_True.json"), 'r')
        for lf in label_dict:
            path_to_labeling_function_eval = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, labeling_function)
            plot_results(path_to_labeling_function_eval, labeling_function=lf, path_to_save=PATH_TO_OUTPUT_DIR, size=16)
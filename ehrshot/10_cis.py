import argparse
import os
import math
import collections
from typing import List
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Make plots/tables of cohort stats")
    parser.add_argument("--path_to_results_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_output_file", required=False, type=str, help="Path to file to save performance results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path_to_results_dir: str = args.path_to_results_dir

    model_heads = [
        # ('clmbr', 'lr_lbfgs'),
        # ('count', 'gbm'),
        # ('count', 'rf'),
        # Debug: Only count, agr, and llm models
        # ('count', 'lr_lbfgs'),
        # ('agr', 'lr_lbfgs'),
        ('llm', 'lr_lbfgs'),
    ]

    # TODO: Better way to do this than hardcoding path
    # Check if in experimental LLM setting, then only consider LLM models
    if '/experiments/' in path_to_results_dir:
        model_heads = [(m, h) for m, h in model_heads if m == 'llm']

    # If output file is provided, create dataframe to save results that can later be written into csv
    results_df = None
    if args.path_to_output_file:
        results_df = pd.DataFrame(columns=['subtask', 'model', 'head', 'score', 'est', 'lower', 'upper'])

    score_map = collections.defaultdict(dict)
    
    LABELING_FUNCTIONS: List[str] = [ x for x in os.listdir(path_to_results_dir) if os.path.isdir(os.path.join(path_to_results_dir, x)) ]
    
    # NOTE: Added selection of tasks that actually finished
    # Choose folder that contain all_results.csv
    LABELING_FUNCTIONS = [ x for x in LABELING_FUNCTIONS if os.path.exists(os.path.join(path_to_results_dir, x, 'all_results.csv')) ]
    
    # Ensure groups are complete or not existent
    def ensure_num_tasks(task: str, num_tasks: int):
        num_tasks_found = len([ x for x in LABELING_FUNCTIONS if x.startswith(task) ])
        assert num_tasks_found == num_tasks or num_tasks_found == 0, f"Expected {num_tasks} tasks for {task}, found {num_tasks_found}"
    ensure_num_tasks('new_', 6)
    ensure_num_tasks('guo_', 3)
    ensure_num_tasks('lab_', 5)
    ensure_num_tasks('chexpert', 1)
    
    SUBTASKS = []

    print("Labeling functions:", LABELING_FUNCTIONS)
    for label in LABELING_FUNCTIONS:
        df = pd.read_csv(os.path.join(path_to_results_dir, label, 'all_results.csv'))
        df = df[df['k'] == -1]
        
        for (model, head) in list(set(list(zip(df['model'], df['head'])))):
            df_ = df[(df['model'] == model) & (df['head'] == head)]
            for idx, row in df_.iterrows():
                if label.startswith('chex'):
                    subtask = 'chex_' + str(row['sub_task'])
                else:
                    subtask = label
                SUBTASKS.append(subtask)
                score_map[(row['score'], subtask, model, head)] = {
                    'score' : float(row['value']),
                    'lower' : float(row['lower']),
                    'upper' : float(row['upper']),
                    'std' : float(row['std']),
                }
    SUBTASKS = list(set(SUBTASKS))

    print("\n\nIndividual Tasks\n\n")
    for score in ['auroc', 'auprc']:
        print("==== SCORE:", score, "====")
        for subtask in SUBTASKS:
            print(subtask)
            for model, head in model_heads:
                v = score_map[(score, subtask, model, head)]
                est = v['score']
                lower = v['lower']
                upper = v['upper']
                print(f"{model}-{head} {est:0.3f} ({lower:0.3f} - {upper:0.3f})")
                if results_df is not None:
                    results_df = results_df._append({'subtask': subtask, 'model': model, 'head': head, 'score': score, 'est': est, 'lower': lower, 'upper': upper}, ignore_index=True)
            print('-' * 80)

    print("\n\nGrouped Tasks\n\n")
    prefixes: List[str] = list({a[:4] for a in LABELING_FUNCTIONS})
    for score in ['auroc', 'auprc']:
        print("==== SCORE:", score, "====")
        for p in prefixes:
            print(p)
            for model, head in model_heads:
                total = 0
                variance = 0
                count = 0
                for subtask in SUBTASKS:
                    if subtask.startswith(p):
                        total += score_map[(score, subtask, model, head)]['score']
                        variance += score_map[(score, subtask, model, head)]['std']**2
                        count += 1
                est = total / count
                std = math.sqrt(variance / count**2)
                lower = est - std * 1.96
                upper = est + std * 1.96
                print(f"{model}-{head} {est:0.3f} ({lower:0.3f} - {upper:0.3f})")
                if results_df is not None:
                    results_df = results_df._append({'subtask': p, 'model': model, 'head': head, 'score': score, 'est': est, 'lower': lower, 'upper': upper}, ignore_index=True)
            print('-' * 80)
            
    if results_df is not None:
        results_df.to_csv(args.path_to_output_file, index=False)
        print(f"Results saved to {args.path_to_output_file}")

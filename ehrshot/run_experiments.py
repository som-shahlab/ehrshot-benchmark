import os
import time
import subprocess
import wandb
import shutil
import argparse
import pandas as pd
from pathlib import Path

def run_command(command):
    """Utility function to run a shell command and print its output."""
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    process.wait()
    
    if process.returncode != 0:
        raise Exception(f"Command failed with error code {process.returncode}")
    return ''.join(output_lines)

def check_slurm_jobs_status(job_ids):
    # Convert list of job IDs to a comma-separated string
    job_ids_str = ','.join(map(str, job_ids))
    
    # Check running or queued jobs using squeue
    squeue_command = f"squeue --jobs={job_ids_str} --noheader --format=%T"
    squeue_output = subprocess.getoutput(squeue_command).splitlines()
    
    return [] if not squeue_output else [status for status in squeue_output]

def main(args):
    # NOTE: Manually skip steps
    start_from_step = 1
    os.chdir(args.base_dir)
    
    # Check that the experiment folder exists
    if not os.path.exists(args.experiment_folder):
        raise ValueError(f"Experiment folder {args.experiment_folder} does not exist")
        
    # Initialize wandb
    experiment_name = Path(args.experiment_folder).name.split("_202")[0]
    # Get experiment identifier from name of folder above
    experiment_id = Path(args.experiment_folder).parent.name
    wandb.init(project=f"ehrshot-{experiment_id}", name=experiment_name)
    
    # Step 1: Generate EHR embeddings
    if start_from_step <= 1:
        tasks_to_instructions = "" if args.task_to_instructions == "" else f"--task_to_instructions {args.task_to_instructions}"
        feature_command = f"""
        python {args.base_dir}/ehrshot/4_generate_llm_features.py \
        --path_to_database {args.path_to_database} \
        --path_to_labels_dir {args.path_to_labels_dir} \
        --num_threads {args.num_threads} \
        --is_force_refresh \
        --path_to_features_dir {args.experiment_folder} \
        --text_encoder {args.text_encoder} \
        --serialization_strategy {args.serialization_strategy} \
        --excluded_ontologies {args.excluded_ontologies} \
        --unique_events {args.unique_events} \
        --numeric_values {args.numeric_values} \
        --medication_entry {args.medication_entry} \
        --num_aggregated {args.num_aggregated} \
        --add_parent_concepts {args.add_parent_concepts} \
        {tasks_to_instructions}
        """
        run_command(feature_command)

    # Step 1.2: Optional - Also evaluate counts and climbr baselines
    # TODO: Unclear if these patient representations remain the same for different patient subgroups (e.g., only new_*)
    # For now just link to representation for all patients via symlinks
    # feature_files = {
    #     'count_features': '/home/sthe14/ehrshot-benchmark/EHRSHOT_ASSETS_old/features/count_features.pkl',
    #     'clmbr_features': '/home/sthe14/ehrshot-benchmark/EHRSHOT_ASSETS_old/features/clmbr_features.pkl',
    #     'agr_features': '/home/sthe14/ehrshot-benchmark/EHRSHOT_ASSETS_old/features/agr_features.pkl',
    # }
    # for feature_name, feature_file in feature_files.items():
    #     feature_symlink = os.path.join(args.experiment_folder, f'{feature_name}.pkl')
    #     if not os.path.exists(feature_symlink):
    #         os.symlink(feature_file, feature_symlink)
    # print(f"Linked {', '.join(list(feature_files.keys()))} features to {args.experiment_folder}")

    # TODO: Not for all - takes too long on GPU node
    # return
            
    # Change into scripts directory
    os.chdir(f"{args.base_dir}/ehrshot/bash_scripts")

    # Step 2: Evaluate embeddings on different tasks
    if start_from_step <= 2:
        eval_script = f"{args.base_dir}/ehrshot/bash_scripts/7_eval.sh"
        eval_command = f"""bash {eval_script} \
        --is_use_slurm \
        --path_to_features_dir {args.experiment_folder} \
        --path_to_output_dir {args.experiment_folder}
        """
        output = run_command(eval_command)

        # Step 2.1: Check for job completion
        job_ids = [int(line.split()[-1]) for line in output.split("\n") if "Submitted batch job" in line]
        print(f"Manual kill command: scancel {' '.join(map(str, job_ids))}")
        status = check_slurm_jobs_status(job_ids)
        while status:
            print(f"Waiting for eval jobs to complete (current status: {[s[0:3] for s in status]})...")
            time.sleep(15)
            status = check_slurm_jobs_status(job_ids)
            
        # Ensure that all subfolder starting with "guo_", "new_", "lab_", "chexpert" have a all_results.csv file
        tasks = ["guo_", "new_", "lab_", "chexpert"]
        for subfolder in os.listdir(args.experiment_folder):
            if any([subfolder.startswith(task) for task in tasks]):
                results_file = os.path.join(args.experiment_folder, subfolder, 'all_results.csv')
                if not os.path.exists(results_file):
                    raise ValueError(f"Results file {results_file} does not exist")

    # Step 3: Calculate metrics
    if start_from_step <= 3:
        calculate_metrics_command = f"""
        python {args.base_dir}/ehrshot/10_cis.py \
        --path_to_results_dir {args.experiment_folder} \
        --path_to_output_file {os.path.join(args.experiment_folder, 'all_results.csv')}
        """
        run_command(calculate_metrics_command)
        
    # Clean up the experiment folder
    llm_features_file = os.path.join(args.experiment_folder, 'llm_features.pkl')
    if os.path.exists(llm_features_file):
        os.remove(llm_features_file)
       
    # Step 4: Log results to wandb
    results_path = os.path.join(args.experiment_folder, 'all_results.csv')
    # Read results and log to wandb
    results = pd.read_csv(results_path)
    # TODO: Use different logging for experiment settings (experimental_setup)
    experimental_setup = {
        "text_encoder": args.text_encoder,
        "serialization_strategy": args.serialization_strategy,
        "add_parent_concepts": args.add_parent_concepts,
        "task_to_instructions": False if args.task_to_instructions == "" else True
    }
    performance_results = {f"{row['subtask']}_{row['score']}": row['est'] for _, row in results.iterrows()}
    wandb.log({**experimental_setup, **performance_results})
    
    # Upload snapshot of key files to wandb
    files_to_upload = [
        f'{args.base_dir}/ehrshot/bash_scripts/4_generate_llm_features.sh',
        f'{args.base_dir}/ehrshot/serialization/text_encoder.py',
        f'{args.base_dir}/ehrshot/serialization/ehr_serializer.py'
    ]
    if args.task_to_instructions:
        files_to_upload.append(args.task_to_instructions)
        
    # Save slurm log file
    slurm_log_file_prefix = f'{args.base_dir}/ehrshot/bash_scripts/logs/ehrshot_'
    slurm_id = os.getenv('SLURM_JOB_ID')
    slurm_log_file = f"{slurm_log_file_prefix}{slurm_id}.log"
    if os.path.exists(slurm_log_file):
        files_to_upload.append(slurm_log_file)
        
    for file_path in files_to_upload:
        shutil.copy(file_path, args.experiment_folder)
        wandb.save(os.path.join(args.experiment_folder, os.path.basename(file_path)))

    print("Experiment completed and results uploaded to wandb.")

    # TODO
    # Sleep for 60 minutes to allow for manual inspection of results
    # time.sleep(3600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EHRShot experiments")
    parser.add_argument("--base_dir", required=True, help="Base directory")
    parser.add_argument("--experiment_folder", required=True, help="Path to the experiment folder")
    parser.add_argument("--path_to_database", required=True, help="Path to the database")
    parser.add_argument("--path_to_labels_dir", required=True, help="Path to the labels directory")
    parser.add_argument("--path_to_split_csv", required=True, help="Path to the CSV file containing splits")
    parser.add_argument("--num_threads", type=int, default=20, help="Number of threads")
    parser.add_argument("--text_encoder", required=True, help="Text encoder to be used")
    parser.add_argument("--serialization_strategy", required=True, help="Serialization strategy to be used")
    parser.add_argument("--excluded_ontologies", type=str, default="", help="Ontologies to exclude")
    parser.add_argument("--unique_events", type=str, default="true", help="Whether to use unique events")
    parser.add_argument("--numeric_values", type=str, default="false", help="Whether to use numeric values")
    parser.add_argument("--medication_entry", type=str, default="false", help="Whether to use a designated medication entry")
    parser.add_argument("--num_aggregated", type=int, default=0, help="Number of aggregated values to use")
    parser.add_argument("--add_parent_concepts", required=True, type=str, help="Category for parent concepts")
    parser.add_argument("--task_to_instructions", type=str, default="", help="Path to task to instructions file")
    
    args = parser.parse_args()
    main(args)
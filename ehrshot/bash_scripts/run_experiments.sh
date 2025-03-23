# Options
num_threads=40
EHRSHOT_ENV="EHRSHOT_ENV"
source activate $EHRSHOT_ENV

# Paths
EXPERIMENT_IDENTIFIER="full_run" # _w_dates
BASE_DIR="/home/sthe14/ehrshot-benchmark"
SCRIPT_DIR="$BASE_DIR/ehrshot"
INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions.json"
# NOTE: Set to experiment path --> only LLM
# NOTE: Set to final_exp path --> LR and CLIMBR as well
EXPERIMENTS_DIR="$BASE_DIR/EHRSHOT_ASSETS/experiments/$EXPERIMENT_IDENTIFIER"
mkdir -p $EXPERIMENTS_DIR

# For neutral instruction ablation - use different instructions file and rename experiment folder to include "_neutr_instr"
# INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions_neutr.json"

# Define the different options to iterate over
text_encoders=(
    "llm2vec_llama3_1_7b_instruct_supervised"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_2k"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_1k"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_512"
    # "llm2vec_llama2_sheared_1_3b_supervised"
    "gteqwen2_7b_instruct"
    # "gteqwen2_7b_instruct_chunked_2k"
    # "gteqwen2_7b_instruct_chunked_1k"
    # "gteqwen2_7b_instruct_chunked_512"
    # "gteqwen2_1_5b_instruct"
    # "bioclinicalbert"
    # "deberta_v3_base"
    # "deberta_v3_large"
    # "bert_base"
    # "bert_large"
    # "modernbert_base"
    # "modernbert_large"
)
# serialization_strategies=(
#     "list_events",
#     "list_visits_with_events",
#     "list_visits_with_events_detailed_aggr"
#     "unique_then_list_visits_wo_allconds_w_values"
#     "unique_then_list_visits_wo_allconds_w_values_4k"
#     "unique_then_list_visits_wo_allconds_w_values_2k"
#     "unique_then_list_visits_wo_allconds_w_values_1k"
#     "unique_then_list_visits_wo_allconds_w_values_512"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_demographics"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_aggregated_events"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_visits"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_conditions"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_medications"
#     "unique_then_list_visits_wo_allconds_w_values_4k_no_procedures"
#     "unique_then_list_visits_wo_allconds_w_values_4k_neutral"
#     "unique_then_list_visits_wo_allconds"
#     "unique_then_list_visits_wo_allconds_4k"
#     "unique_then_list_visits_w_values"
#     "unique_then_list_visits_w_values_4k" # <- llama
#     "unique_then_list_visits"
#     "unique_then_list_visits_4k"
# )

# Selected serialization and settings
serialization_strategies=("unique_then_list_visits_wo_allconds_w_values_4k")

# Fixed options
instructions_options=("true")
excluded_ontologies=("no_labs_single")
num_aggregated=(3)
time_window_days=(0)
# time_window_days=(0 1 7 30 365)

# Labels = Dataset subset
DATASET="full"

if [ $DATASET == "full" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark
elif [ $DATASET == "new_guo_chexpert" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark_subsets/new_guo_chexpert # new_guo_chexpert
elif [ $DATASET == "new_guo" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark_subsets/new_guo # new_guo
fi

for text_encoder in "${text_encoders[@]}"; do
    for serialization_strategy in "${serialization_strategies[@]}"; do
        for excluded_ontology in "${excluded_ontologies[@]}"; do
            for num_aggregated_val in "${num_aggregated[@]}"; do
                for time_window_days_val in "${time_window_days[@]}"; do
                    for use_instructions in "${instructions_options[@]}"; do
                        # Define experiment name based on concatenation of options and create timestamped directory
                        instructions_suffix="_no_instr"
                        instructions_file_arg=""
                        if [ $use_instructions == "true" ]; then
                            instructions_suffix=""
                            instructions_file_arg="$INSTRUCTIONS_FILE"
                        fi

                        experiment_name="${text_encoder}_${serialization_strategy}_${excluded_ontology}_${num_aggregated_val}_${time_window_days_val}_${DATASET}${instructions_suffix}"
                        experiment_dir="${EXPERIMENTS_DIR}/${experiment_name}"
                        mkdir -p $experiment_dir

                        # Check if the experiment has already been run by testing if all_results.csv exists in the experiment directory
                        if [ -f "${experiment_dir}/all_results.csv" ]; then
                            echo "Experiment $experiment_name already exists. Skipping..."
                            continue
                        fi

                        # Run the experiment with bash or slurm
                        cmd="bash"
                        [[ " $* " == *" --is_use_slurm "* ]] && cmd="sbatch"

                        $cmd /home/sthe14/ehrshot-benchmark/ehrshot/bash_scripts/run_experiments_helper.sh \
                            $BASE_DIR \
                            $experiment_dir \
                            $BASE_DIR/EHRSHOT_ASSETS/femr/extract \
                            $LABELS_DIR \
                            $BASE_DIR/EHRSHOT_ASSETS/splits/person_id_map.csv \
                            $num_threads \
                            $text_encoder \
                            $serialization_strategy \
                            $excluded_ontology \
                            $num_aggregated_val \
                            $time_window_days_val \
                            $instructions_file_arg
                    done
                done
            done
        done
    done
done
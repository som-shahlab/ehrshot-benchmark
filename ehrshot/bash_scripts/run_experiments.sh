# Options
num_threads=40
EHRSHOT_ENV="EHRSHOT_ENV"
source activate $EHRSHOT_ENV

# Paths
EXPERIMENT_IDENTIFIER="markdown_serialization_baselines" # _w_dates
BASE_DIR="/home/sthe14/ehrshot-benchmark"
SCRIPT_DIR="$BASE_DIR/ehrshot"
INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions.json"
# NOTE: Set to experiment path --> only LLM
# NOTE: Set to final_exp path --> LR and CLIMBR as well
EXPERIMENTS_DIR="$BASE_DIR/EHRSHOT_ASSETS/experiments/$EXPERIMENT_IDENTIFIER"
mkdir -p $EXPERIMENTS_DIR

# For now: always use instructions and no parent concepts
instructions_options=("true")
parent_concepts=("none")

# Define the different options to iterate over
text_encoders=(
    "llm2vec_llama3_1_7b_instruct_supervised"
    # llm2vec_llama3_1_7b_instruct_supervised_chunked_2k
    # llm2vec_llama3_1_7b_instruct_supervised_chunked_1k
    # llm2vec_llama3_1_7b_instruct_supervised_chunked_512
    # "llm2vec_llama2_sheared_1_3b_supervised"
    # "gteqwen2_7b_instruct"
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
#     "unique_then_list_visits_wo_allconds"
#     "unique_then_list_visits_wo_allconds_4k"
#     "unique_then_list_visits_w_values"
#     "unique_then_list_visits_w_values_4k"
#     "unique_then_list_visits"
#     "unique_then_list_visits_4k"
# )
# excluded_ontologies=("none" "no_labs" "no_labs_meds" "no_labs_meds_single" "no_labs_single")
# unique_events=("true" "false")
# numeric_values=("true" "false")
# num_aggregated=(0 1 3 5)
# parent_concepts=("none")

# Selected serialization and settings
serialization_strategies=("unique_then_list_visits_wo_allconds_w_values_4k")

excluded_ontologies=("no_labs_single")
unique_events=("true")
numeric_values=("false")
medication_entry=("true")
num_aggregated=(3)

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
            for unique_event in "${unique_events[@]}"; do
                for numeric_value in "${numeric_values[@]}"; do
                    for medication_entry_val in "${medication_entry[@]}"; do
                        for num_aggregated_val in "${num_aggregated[@]}"; do
                            for parent_concept in "${parent_concepts[@]}"; do
                                for use_instructions in "${instructions_options[@]}"; do
                                    # Define experiment name based on concatenation of options and create timestamped directory
                                    instructions_suffix=""
                                    instructions_file_arg=""
                                    if [ $use_instructions == "true" ]; then
                                        instructions_suffix="_instr"
                                        instructions_file_arg="$INSTRUCTIONS_FILE"
                                    fi

                                    # Ommit _${parent_concept}${instructions_suffix} as always the same for now
                                    experiment_name="${text_encoder}_${serialization_strategy}_${excluded_ontology}_${unique_event}_${numeric_value}_${medication_entry_val}_${num_aggregated_val}_${DATASET}"
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
                                        $unique_event \
                                        $numeric_value \
                                        $medication_entry_val \
                                        $num_aggregated_val \
                                        $parent_concept \
                                        $instructions_file_arg
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
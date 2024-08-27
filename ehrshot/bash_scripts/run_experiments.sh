# Options
num_threads=20
EHRSHOT_ENV="EHRSHOT_ENV"
source activate $EHRSHOT_ENV

# Paths
EXPERIMENT_IDENTIFIER="long_context_and_instructions"
BASE_DIR="/home/sthe14/ehrshot-benchmark"
SCRIPT_DIR="$BASE_DIR/ehrshot"
INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions.json"
EXPERIMENTS_DIR="$BASE_DIR/EHRSHOT_ASSETS/experiments/$EXPERIMENT_IDENTIFIER"
mkdir -p $EXPERIMENTS_DIR

# Define the different options to iterate over
text_encoders=(
    "llm2vec_llama3_7b_instruct_supervised"
    "llm2vec_llama3_1_7b_instruct_supervised"
    "gteqwen2_7b_instruct"
    "st_gte_large_en_v15"
    "bioclinicalbert-fl"
    "bioclinicalbert-fl-average-chunks"
    "longformerlarge-fl"
    "biomedicallongformerlarge-fl"
)
serialization_strategies=(
    "list_unique_events_wo_numeric_values"
    "list_visits_with_events_wo_numeric_values"
    "list_visits_with_events"
)
instructions_options=("true" "false")

for text_encoder in "${text_encoders[@]}"; do
    for serialization_strategy in "${serialization_strategies[@]}"; do
        for use_instructions in "${instructions_options[@]}"; do
            # Define experiment name based on concatenation of options and create timestamped directory
            instructions_suffix=""
            instructions_file_arg=""
            if [ $use_instructions == "true" ]; then
                instructions_suffix="_instr"
                instructions_file_arg="$INSTRUCTIONS_FILE"
            fi
            experiment_name="${text_encoder}_${serialization_strategy}${instructions_suffix}"
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
                $BASE_DIR/EHRSHOT_ASSETS/benchmark \
                $BASE_DIR/EHRSHOT_ASSETS/splits/person_id_map.csv \
                $num_threads \
                $text_encoder \
                $serialization_strategy \
                $instructions_file_arg
        done
    done
done
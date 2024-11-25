# Options
num_threads=20
EHRSHOT_ENV="EHRSHOT_ENV"
source activate $EHRSHOT_ENV

# Paths
EXPERIMENT_IDENTIFIER="markdown_serialization"
BASE_DIR="/home/sthe14/ehrshot-benchmark"
SCRIPT_DIR="$BASE_DIR/ehrshot"
INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions.json"
EXPERIMENTS_DIR="$BASE_DIR/EHRSHOT_ASSETS/experiments/$EXPERIMENT_IDENTIFIER"
mkdir -p $EXPERIMENTS_DIR

# Define the different options to iterate over
text_encoders=(
    # "llm2vec_llama3_7b_instruct_supervised"
    # "llm2vec_llama3_1_7b_instruct_supervised"
    # "llm2vec_mistral_7b_instruct_supervised"
    # "gteqwen2_7b_instruct"
    # "gteqwen2_1_5b_instruct"
    # "st_gte_large_en_v15"
    # "bioclinicalbert-fl"
    # "bioclinicalbert-fl-average-chunks"
    # "longformerlarge-fl"
    # "biomedicallongformerlarge-fl"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-simcse-mimic/Meta-Llama-3.1-8B-Instruct_2000_mntp_steps_02/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-simcse-mimic/Meta-Llama-3.1-8B-Instruct_2000_mntp_steps_03/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-simcse-mimic/Meta-Llama-3.1-8B-Instruct_2000_mntp_steps_04/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-simcse-mimic/Meta-Llama-3.1-8B-Instruct_2000_mntp_steps_05/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps_2e-4/MimicIVDISup_train_m-Meta-Llama-3.1-8B-Instruct_l-2048/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/MedNLI_train_m-Meta-Llama-3.1-8B-Instruct_l-512/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic/Meta-Llama-3.1-8B-Instruct/MimicIVDISup_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-sup/Meta-Llama-3.1-8B-Instruct/MimicIVDISup_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mednli/Meta-Llama-3.1-8B-Instruct/MedNLI_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mednli-sup/Meta-Llama-3.1-8B-Instruct/MedNLI_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct/checkpoint-1000"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.1_medical/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.1_medical/checkpoint-400"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.1_medical/checkpoint-1000"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical/checkpoint-200"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical/checkpoint-400"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical/checkpoint-1000"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical_2000/checkpoint-400"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical_2000/checkpoint-1000"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical_2000/checkpoint-1400"
    # "llm2vec_llama3_1_7b_instruct_mimic_/mntp-supervised-mimic-mednli/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5MM_train_m-Meta-Llama-3.1-8B-Instruct_0.2_medical_2000/checkpoint-2000"
)
serialization_strategies=(
    "list_unique_events_wo_numeric_values"
    # "list_visits_with_events_wo_numeric_values"
    # "list_visits_with_events"
    # "list_visits_with_unique_events_wo_numeric_values"
    # "list_visits_with_unique_events"
)
excluded_ontologies=("none" "no_labs" "no_labs_meds")
parent_concepts=("none" "conditions")
instructions_options=("true" "false")

for text_encoder in "${text_encoders[@]}"; do
    for serialization_strategy in "${serialization_strategies[@]}"; do
        for excluded_ontology in "${excluded_ontologies}[@]}"; do
            for parent_concept in "${parent_concepts[@]}"; do
                for use_instructions in "${instructions_options[@]}"; do
                    # Define experiment name based on concatenation of options and create timestamped directory
                    instructions_suffix=""
                    instructions_file_arg=""
                    if [ $use_instructions == "true" ]; then
                        instructions_suffix="_instr"
                        instructions_file_arg="$INSTRUCTIONS_FILE"
                    fi

                    experiment_name="${text_encoder}_${serialization_strategy}_${excluded_ontology}_${parent_concept}${instructions_suffix}"
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
                        $excluded_ontology \
                        $parent_concept \
                        $instructions_file_arg
                done
            done
        done
    done
done
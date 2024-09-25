#!/bin/bash

# Usage: 
#     bash stratify.sh

labeling_functions=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    # "chexpert" 
)

for labeling_function in "${labeling_functions[@]}"
do
    echo "----------------------------------------"
    echo $labeling_function
    echo "START | Current time: $(date +"%T")"
    python3 stratify.py --task $labeling_function
    echo "END | Current time: $(date +"%T")"
done
import os
import pickle

PATH_TO_DATA="/share/pi/nigam/rthapa84/data/MLHC/cohort/good-cohort/deterministic_experiment_top_100_based_on_counts"
labeling_function = 'guo_los'
PATH_TO_SAVE_SHOTS: str = os.path.join(PATH_TO_DATA, f"few_shot_train_test_eval_data/{labeling_function}")
PATH_TO_SAVE = os.path.join(PATH_TO_SAVE_SHOTS, "few_shots_data.pickle")

few_shots_dict = pickle.load(open(PATH_TO_SAVE, 'rb'))

FEW_SHOTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128]
MODELS = ['Codes_Only', 'Count_based_GBM']

for k in FEW_SHOTS:
    # Check that lengths are correct
    for model in MODELS:
        for split in [ 'X_train_k', 'y_train_k', 'X_val_k', 'y_val_k', ]:
            for replicate in range(5):
                n_rows = few_shots_dict[labeling_function][k][replicate][model][split].shape[0]
                assert k * 2 == n_rows, f"{split}, {k}, {n_rows}"
    assert k * 2 == len(few_shots_dict[labeling_function][k][replicate]['patient_ids_train_k'])
    assert k * 2 == len(few_shots_dict[labeling_function][k][replicate]['label_times_train_k'])
    assert k * 2 == len(few_shots_dict[labeling_function][k][replicate]['patient_ids_val_k'])
    assert k * 2 == len(few_shots_dict[labeling_function][k][replicate]['label_times_val_k'])

    # Check that val/test Y are same for each model
    assert (few_shots_dict[labeling_function][k][replicate][MODELS[0]]['y_val_k'] == few_shots_dict[labeling_function][k][replicate][MODELS[1]]['y_val_k']).all()
    assert (few_shots_dict[labeling_function][k][replicate][MODELS[0]]['y_test_k'] == few_shots_dict[labeling_function][k][replicate][MODELS[1]]['y_test_k']).all()
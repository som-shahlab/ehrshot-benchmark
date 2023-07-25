import json
import os

for label_name in os.listdir('EHRSHOT_ASSETS/labels'):
    source_dir = os.path.join('../old_assets/benchmark', label_name)
    target_dir = os.path.join('EHRSHOT_ASSETS/labels/', label_name)

    if label_name == 'chexpert':
        continue

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    source_file = os.path.join(source_dir, 'few_shots_data.json')
    target_file = os.path.join(target_dir, 'few_shots_data.json')

    with open(source_file) as source:
        source_data = json.load(source)[label_name]

    for v in source_data.values():
        for v2 in v.values():
            del v2['train_idxs']
            del v2['val_idxs']

    with open(target_file, 'w') as target:
        json.dump(source_data, target)
import csv
import datetime
import os

for label_name in os.listdir('EHRSHOT_ASSETS/old_labels'):
    source_dir = os.path.join('EHRSHOT_ASSETS/old_labels/', label_name)
    target_dir = os.path.join('EHRSHOT_ASSETS/labels/', label_name)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    source_file = os.path.join(source_dir, 'labeled_patients.csv')
    target_file = os.path.join(target_dir, 'labeled_patients.csv')

    with open(source_file) as source:
        with open(target_file, 'w') as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                prediction_time = datetime.datetime.fromisoformat(row['prediction_time'])
                fixed_prediction_time = prediction_time.replace(second=0)
                row['prediction_time'] = fixed_prediction_time.isoformat()

                writer.writerow(row)
    
import csv
import json

csv_path = 'EHRSHOT_ASSETS/splits/person_id_map.csv'
json_path = 'EHRSHOT_ASSETS/splits/splits.json'

# Read the csv 
with open(csv_path, 'r') as f:
    csv_data = f.read()

# Convert the CSV string to a list of dictionaries
csv_lines = csv_data.strip().split("\n")
reader = csv.DictReader(csv_lines)

# Initialize a dictionary to hold the JSON structure
json_data = {}

# Process the CSV data
for row in reader:
    key = row['split']
    value = int(row['omop_person_id'])
    if key not in json_data:
        json_data[key] = []
    json_data[key].append(value)

# Write the JSON data to a file
print(f"Writing JSON data to {json_path}")
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
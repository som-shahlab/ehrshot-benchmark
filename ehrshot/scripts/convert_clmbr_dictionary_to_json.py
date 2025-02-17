"""Run this script to view the vocabulary of the CLMBR model in human-readable format.

This works by converting `EHRSHOT_ASSETS/data/dictionary` (which is a .msgpack file) into three .json files:
    * dictionary.json => Raw representation of the exact contents in dictionary (which is a .msgpack file)
    * token_2_code.json => Dictionary where [key] = token ID (e.g. '0'), [value] = code (e.g. 'SNOMED/3950001')
    * token_2_description.json => Dictionary where [key] = token ID (e.g. '0'), [value] = code (e.g. 'Birth')
"""

import msgpack
import json
import femr.datasets

femr_db = femr.datasets.PatientDatabase('../../EHRSHOT_ASSETS/femr/extract')

# Read the msgpack file
with open('../../EHRSHOT_ASSETS/models/clmbr/dictionary', 'rb') as msgpack_file:
    data = msgpack.unpack(msgpack_file)

# Raw msgpack converted to JSON
with open('../../EHRSHOT_ASSETS/models/clmbr/dictionary.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)

# Human-friendly token => code / description mapping
with open('../../EHRSHOT_ASSETS/models/clmbr/token_2_code.json', 'w') as json_file:
    json.dump({ idx: val['code_string'] for idx, val in enumerate(data['regular']) }, json_file, indent=2)
with open('../../EHRSHOT_ASSETS/models/clmbr/token_2_description.json', 'w') as json_file:
    mapping = {}
    for idx, val in enumerate(data['regular']):
        try:
            mapping[idx] = femr_db.get_ontology().get_text_description(val["code_string"])
        except:
            mapping[idx] = None
    json.dump(mapping, json_file, indent=2)

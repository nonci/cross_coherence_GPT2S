#51,912c044ef5e129522c98a1adaac88b94,"A rectangular grey...",Table,4379243,3982430,text_51.pt

import csv, json, os, sys
from collections import OrderedDict
from inspect import getsourcefile
sys.path.append(os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-2]))
from cross_coherence_GPT2S.my_config import get_config
config = get_config()

CATEGORIES_FROM = f'{config.t2s_dataset}/annotations_chatgpt/from_shapenet_v1/test_all.csv'
TARGET_JSON = f'{config.base_dir}/data/human_test_set_final.json'
NEW_JSON = f'{TARGET_JSON.rpartition(".")[0]}_with_cats.json'
PRETTY_PRINT = True  # Set to False for slightly smaller file
WRITE = False

categories = {'chair':'03001627', 'table':'04379243'}
id_to_cat = OrderedDict()

with open(CATEGORIES_FROM, "r") as f:
	reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, strict=True, )
	for row in reader: break
	#0,modelId,description,category,topLevelSynsetId,subSynsetId,tensor
	for row in reader:
		id_to_cat[row[1]] = int(row[4])

with open(TARGET_JSON, 'rb') as f:
    d = json.load(f)

hard, easy, hard_and_diff_cat = 0, 0, 0
# Update with category and statistics
for row in d:
    hard += row['task'] == 'hard'
    easy += row['task'] == 'easy'
    hard_and_diff_cat += (id_to_cat[row['dist_id']] != id_to_cat[row['gt_id']]) and row['task']=='hard'
    row['dist_cat'] = id_to_cat[row['dist_id']]
    row['gt_cat'] = id_to_cat[row['gt_id']]

print(f'hard: {hard} (of which {hard_and_diff_cat} of differ. cat.), easy: {easy}')
print('keys are:', ', '.join(row.keys()))

if WRITE:
	assert(not os.path.exists(NEW_JSON))
	# Writing
	with open(NEW_JSON, 'w') as f:
		json.dump(d, f, indent=4 if PRETTY_PRINT else None)
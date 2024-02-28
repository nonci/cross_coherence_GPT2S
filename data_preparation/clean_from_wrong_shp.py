# Delete from file wrong shapes
# call with <in_fname>.csv as 1st par, <out_fname>.csv as 2nd; for example:
# python3 clean_from_wrong_shp.py /.../datasets/Text2Shape/annotations_chatgpt/from_text2shape/train_all.csv /.../datasets/Text2Shape/train_all_clean.csv

import csv, sys, os
from inspect import getsourcefile
sys.path.append(os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-2]))
from my_config import get_config
config = get_config()

good_path = f'{config.t2s_dataset}/shapes/text2shape_rgb'
search_path = f'{config.t2s_dataset}/shapes'
disc = 0

# Building file dict in search path:
#paths = collections.defaultdict(lambda:0)
all_files = dict()
for dirpath, _, files in os.walk(search_path):
    for file in files:
        all_files[file] = dirpath

with open(sys.argv[1], 'r') as ifile, open(sys.argv[2], 'w') as outfile:
    reader = csv.reader(ifile, delimiter=',', quotechar='"')
    writer = csv.writer(outfile, delimiter=',', quotechar='"')
    writer.writerow(next(reader)) # header
    for row in reader:
        fname = f'{row[1]}.ply'
        if all_files[fname] == good_path:
            writer.writerow(row)
        else:
            disc += 1
print(disc, 'discarded')
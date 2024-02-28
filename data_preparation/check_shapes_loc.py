# SANITY CHECK
# Checks the path of the shapes in the given .csv file
# 
# call with <fname>.cvs as 1st par
from inspect import getsourcefile
import sys, os
from my_config import get_config
sys.path.append(os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-2]))
config = get_config()

search_path = f'{config.t2s_dataset}/shapes'

import csv, sys, os, collections
paths = collections.defaultdict(lambda:0)
all_files = dict()
for dirpath, _, files in os.walk(search_path):
    for file in files:
        all_files[file] = dirpath

with open(sys.argv[1], 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',', quotechar='"')
    _ = next(reader) # skip header
    for row in reader:
        fname = f'{row[1]}.ply'
        paths[all_files[fname]] += 1
            
print('Shapes are in:')
for k in paths:
    print(f'  {k} ({paths[k]} occurrences)')
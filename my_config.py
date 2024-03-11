import os
import json
from inspect import getsourcefile

# portable and wd-indep. way:
this_folder = os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-1]+[''])

class Config:
    def __init__(self, fname) -> None:
        with open(fname, 'r') as f: c = json.load(f)
        self.base_dir = c['base_dir']
        self.clouds_dir = c['clouds_dir']
        self.t2s_dataset = c['t2s_dataset']
        self.render_dir = c['render_dir']
        self.render_dir2 = c['render_dir2']
        self.text_emb_dir = c['text_emb_dir']
        self.shapenet_partseg = c['shapenet_partseg']
        self.t2s_csvs = c['t2s_csvs']
		
		
def get_config(config_file=f'{this_folder}config.json'):
    if not os.path.exists(config_file):
        print('Please specify a config json file.')
        raise FileNotFoundError
    return Config(config_file)
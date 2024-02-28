'''
Most of the original file removed.
'''

import json
import sys
sys.path.append("../Infusion")
from data.text2shape_dataset import Text2Shape, Text2Shape_pairs, Text2Shape_pairs_easy_hard, Text2Shape_humaneval, visualize_data_sample, visualize_data_sample_color, shuffle_ids

from pathlib import Path
from tqdm import tqdm

from my_config import get_config
config = get_config()


def build_mids_texts_tensornames(json_path, device):
    '''
        this function builds a list of dictionaries containing UNIQUE pairs of mids and texts, where mid = gt_id from the JSON file.
    '''
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    mids_texts = []
    
    # instantiate test set of Text2Shape, to extract category
    test_ds =   Text2Shape(
                            root=Path(config.t2s_dataset),
                            chatgpt_prompts=True,
                            split='test',
                            categories='all',
                            from_shapenet_v1=False,
                            from_shapenet_v2=False,
                            language_model="t5-11b",
                            lowercase_text=True,
                            max_length=77,
                            padding=False,
                            conditional_setup=True,
                            scale_mode="shapenet_v1_norm")
    
    mid_cate = {}   # dictionary which associates category to gt_mid
    for test_sample in test_ds:
        gt_mid = test_sample["model_id"]
        cate = test_sample["cate"]
        mid_cate[gt_mid] = cate
        
    print('mid_cate len: ', len(mid_cate))
    
    for idx, d in enumerate(tqdm(json_data, desc='Parsing JSON data...')):
        mid = d["gt_id"]   
        cate = mid_cate[mid]
        text = d["text"]
        tensor_name = d["tensor_name"]
        dataset = d["dataset"]
        mids_texts.append(
            {
            "mid": mid,
            "text": text,
            "tensor_name": tensor_name,
            "dataset": dataset,
            "cate": cate
            })
    
    # Remove repetitions of combinations of mid and text
    new_list = []
    for d in mids_texts:    # len of mids_texts: 2153
        # Check if the dictionary has a unique pair of "mid" and "text" values
        if all(d['mid'] != x['mid'] or d['text'] != x['text'] or d['tensor_name'] != x['tensor_name'] or d['dataset'] != x['dataset'] for x in new_list):
            # If the pair is unique, add the dictionary to the new list
            new_list.append(d)

    return new_list         # len: 1749
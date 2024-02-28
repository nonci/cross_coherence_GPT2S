import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Callable
from tqdm import tqdm

import sys
from data.text2shape_dataset import Text2Shape


class ShapeglotDataset(Dataset):
    def __init__(self, np_data, shuffle_geo=False, target_last=False):
        """
        :param np_data:
        :param shuffle_geo: if True, the positions of the shapes in context are randomly swapped.
        """
        super(ShapeglotDataset, self).__init__()
        self.data = np_data
        self.shuffle_geo = shuffle_geo
        self.target_last = target_last

    def __getitem__(self, index):
        text = self.data['text'][index].astype(np.long)
        geos = self.data['in_geo'][index].astype(np.long)
        target = self.data['target'][index]
        idx = np.arange(len(geos))

        if self.shuffle_geo:
            np.random.shuffle(idx)

        geos = geos[idx]
        target = np.argmax(target[idx])

        if self.target_last:
            last = geos[-1]
            geos[-1] = geos[target]
            geos[target] = last
            target = len(geos) - 1

        return geos, target, text

    def __len__(self):
        return len(self.data)

class T2SShapeglotDataset(Dataset):
    def __init__(
    self, root, split, categories, from_shapenet_v1, from_shapenet_v2, 
    language_model, lowercase_text, max_length, padding, conditional_setup, scale_mode, shuffle_pairs):
        
        super(T2SShapeglotDataset, self).__init__()

        '''
        Class implementing the dataset to train ShapeGlot, based on Text2Shape dataset.
        For each model_id, we pick one other model_id to build a pair of shapes.
        We repeat this twice.
        Reference: shapenet_chairs.csv
        Each pair contains:
            - 2 pcs (model_ids or tensors of pts)
            - text embedding of the sentence
            - index of the target shape (will be always the 1st one)
        When using categories "all", I can have pairs with some tables and chairs
        '''

        # Read T2S dataset
        # iterate over dataset:
            # build list of dictionaries with 3 clouds, a text embed and target idx
        
        t2s_ds = Text2Shape(root, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, max_length, padding, conditional_setup, scale_mode)

        self.shuffle_pairs = shuffle_pairs
        self.pairs = []
        for i in tqdm(range(0, len(t2s_ds))):
            # for each element I pick one other random model_id WITHIN CATEGORY => I get pairs of chairs together and pairs of tables together
            shape_class = t2s_ds[i]["cate"]
            target_model_id = t2s_ds[i]["model_id"]
            # pick randomly another chair
            t2s_cates = np.array([data["cate"] for data in t2s_ds])     # we need np array to use np.where
            t2s_ids = np.array([data["model_id"] for data in t2s_ds])    # we need np array to use np.where
            idxs = np.where((t2s_cates == shape_class) & (t2s_ids != target_model_id))[0]
            chosen_idx = np.random.choice(idxs) # choose a random idx among the accepted ones
            distr_model_id = t2s_ds[chosen_idx]["model_id"]
            text_embed = t2s_ds[i]["text"]    # get text embedding of the current sentence

            model_ids = [target_model_id, distr_model_id]
            target = 0
            # shuffle the order of the model_ids and the corresponding target
            if self.shuffle_pairs:
                model_ids, target = shuffle_ids(model_ids, target)
    
            self.pairs.append({
                "model_ids": model_ids,
                "target": target,
                "text_embed": text_embed
            })

        # REMARK: I can have the same pair of model_ids multiple times, with different text embeddings => CORRECT!
        print('all pairs are ready')


    def __getitem__(self, index):
        model_ids = self.pairs[index]["model_ids"]
        target = self.pairs[index]["target"]
        text_embed = self.pairs[index]["text_embed"]

        return model_ids, text_embed, target

    def __len__(self):
        return len(self.pairs)
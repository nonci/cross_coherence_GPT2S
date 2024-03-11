from curses.textpad import Textbox
from typing import List, Callable
from pathlib import Path
from torch.utils.data import Dataset
from pycarus.transforms.var import Compose
import pandas as pd
from pytorch3d.io import IO
import torch
from copy import copy as shallow_copy
from copy import deepcopy
import numpy as np
import os
import random
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json, sys
from inspect import getsourcefile
sys.path.append(os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-2]))
from my_config import get_config
config = get_config()

DEFAULT_T5_NAME = 't5-11b'

def shuffle_ids(ids, label, random_seed=None):
    ''' e.g. if [a, b] with label 0 makes it [b, a] with label 1.
    '''
    res_ids = ids.copy()   # initialization of output list
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle = np.random.shuffle
    idx = np.arange(0, len(ids))
    shuffle(idx)
    # if idx==0, no swap
    # if idx==1, swap elements of ids
    i=0
    for one_idx in idx:
        res_ids[i] = ids[one_idx]
        i += 1 
    target = idx[0]

    return res_ids, target

def visualize_data_sample(pointclouds, target, img_title, path, idx):
    n_clouds = len(pointclouds)
    fig = plt.figure(figsize=(10,10))
    wrapped_title = "\n".join(img_title[i:i+50] for i in range(0, len(img_title), 50))
    plt.title(label=wrapped_title, fontsize=15)
    plt.axis('off')
    ncols = n_clouds
    nrows = 1
    for idx, pc in enumerate(pointclouds):
        if pc.shape[1]==6:
            colour = pc[:, 3:]
        else:
            if isinstance(target, int):
                colour = 'r' if target == idx else 'b'
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=10)
        ax.view_init(elev=30, azim=255)
        ax.set_facecolor("grey")
        ax.axis('off')
        if isinstance(target, List):
            ax.title.set_text(target[idx])
            ax.title.set_fontsize(16)
    plt.savefig(path)
    plt.close(fig)

def visualize_data_sample_color(target_mid, dist_mid, img_title, path, idx):
    fig = plt.figure(figsize=(20,20))
    plt.title(label=img_title, fontsize=25)
    plt.axis('off')
    nrows = 1
    # plot target
    ax = fig.add_subplot(nrows, 2, 1)
    ax.axis('off')
    img_path = os.path.join(f'{config.t2s_dataset}/nrrd_256_filter_div_128_solid/', target_mid, target_mid + '.png')
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.title.set_text('Target')

    # plot dist
    ax = fig.add_subplot(nrows, 2, 2)
    ax.axis('off')
    img_path = os.path.join(f'{config.t2s_dataset}/nrrd_256_filter_div_128_solid/', dist_mid, dist_mid + '.png')
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.title.set_text('Distractor')

    plt.savefig(path)
    plt.close(fig)


class Text2Shape(Dataset):
    def __init__(
        self,
        root: Path,
        chatgpt_prompts: bool,
        split: str,
        categories: str,
        from_shapenet_v1: bool,
        from_shapenet_v2: bool,
        language_model: str,
        lowercase_text: bool,
        max_length: int,
        padding: bool,
        conditional_setup: bool,
        scale_mode: str,
        transforms: List[Callable] = []
    ) -> None:

        '''
        Class implementing the Text2Shape dataset proposed in:

            Chen, Kevin and Choy, Christopher B and Savva, Manolis and Chang, Angel X and Funkhouser, 
            Thomas and Savarese, Silvio (2019).
            Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings
            arXiv preprint arXiv:1803.08495

            Args:
                root: The path to the folder containing the dataset.
                split: The name of the split to load.
                categories: A list of shape categories to select. Defaults to [].
                transforms_complete: The transform to apply to the complete cloud. Defaults to [].
                transforms_incomplete: The transform to apply to the incomplete cloud. Defaults to [].
                transforms_all: The transform to apply to both the point clouds. Defaults to [].

            Raises:
                FileNotFoundError: If the folder does not exist.
                ValueError: If the chosen split is not allowed.
            '''
        super().__init__()
        print('initializing dataset...')
        self.split = split
        self.splits = ["train", "val", "test"]
        if self.split not in self.splits:
            raise ValueError(f"{self.split} value not allowed, only allowed {self.splits}.")
        self.root = root # dataset_text2shape
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} not found.")
        self.transform = Compose(transforms)
        self.categories = categories
        self.scale_mode = scale_mode
        
        self.pointclouds=[]
        
        # get text_prompt, model_id, category
        if from_shapenet_v2:
            if chatgpt_prompts:
                annotations_path = self.root / "annotations_chatgpt" / "from_shapenet_v2" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations_chatgpt" / "from_shapenet_v2" / f"train_{self.categories}.csv"
            else:
                annotations_path = self.root / "annotations" / "from_shapenet_v2" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations" / "from_shapenet_v2" / f"train_{self.categories}.csv"
        elif from_shapenet_v1:
            if chatgpt_prompts:
                annotations_path = self.root / "annotations_chatgpt" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations_chatgpt" / "from_shapenet_v1" / f"train_{self.categories}.csv"
            else:
                annotations_path = self.root / "annotations" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations" / "from_shapenet_v1" / f"train_{self.categories}.csv"                
        elif chatgpt_prompts:
            annotations_path = self.root / "annotations_chatgpt" / "from_text2shape" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations_chatgpt" / "from_text2shape" / f"train_{self.categories}.csv"
        else:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations" / "from_text2shape" / f"train_{self.categories}.csv"


        print('annotations: ', annotations_path)
        df = pd.read_csv(annotations_path, quotechar='"')
        # if I am training only with shapes, I remove rows of the same shape and different text
        if not conditional_setup:
            df.drop_duplicates(subset=['modelId'], inplace=True)
        
        if self.scale_mode == 'global_unit':   # normalize with mean and std of training set of Text2Shape
            # read DataFrame of training set
            train_ds_df = pd.read_csv(train_path, quotechar='"')
            train_ds_df.drop_duplicates(subset=['modelId'], inplace=True)
            global_mean, global_std = self.get_ds_statistics(train_ds_df, from_shapenet_v1, from_shapenet_v2)
            print('normalization with mean and std of training set')
            print('global mean: ', global_mean)
            print('global std: ', global_std)

        count_trunc = 0
        for idx, row in df.iterrows():
            # read_pcd
            text = row["description"]
            text = text.strip('"')
            model_id = row["modelId"]
            category = row["category"]
            if from_shapenet_v1:
                model_path = str(self.root / "shapes" / "shapenet_v1" / f"{model_id}.ply")           
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
                
            elif from_shapenet_v2:
                model_path = str(self.root / "shapes" / "shapenet_v2" / f"{model_id}.ply")
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
            else:
                model_path = str(self.root / "shapes" / "text2shape_rgb" / f"{model_id}.ply")
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                
                if points.shape[0] != 2048:  # happens for model_id: 791c14d53bd565f56ba14bfd91a75020 (empty) and others
                    continue
                
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
                
            #scaling the pcd (from diffusion-pointcloud code)
            if self.scale_mode == "global_unit":
                shift = global_mean
                scale = global_std
            elif self.scale_mode == 'shape_unit':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            elif self.scale_mode == 'shapenet_v1_norm':
                    pc_max, _ = pc[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = torch.linalg.norm(pc_max - pc_min).reshape(1, 1)
            else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])
            pc[:,:3] = (pc[:,:3] - shift) / scale

            tensor_name = row["tensor"]
            if lowercase_text:
                if padding:
                    if chatgpt_prompts:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                    else:
                        text_embed_path = self.root / "text_embeds" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                if padding:
                    if chatgpt_prompts:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" /  "padding" / f"{tensor_name}"
                    else:
                        text_embed_path = self.root / "text_embeds" / language_model / "std" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "std" / f"{tensor_name}"
            
            if padding:
                text_embed = torch.load(text_embed_path, map_location='cpu')
            else:
                text_embed = torch.load(text_embed_path, map_location='cpu')
                if text_embed.shape[0] > max_length:
                    count_trunc += 1
                    text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
                
            # build mask for this text embedding (i have text embeds length and max length)
            #key_padding_mask_false = torch.zeros((1+text_embed.shape[0]), dtype=torch.bool)             # False => elements will be processed
            #key_padding_mask_true = torch.ones((max_length - text_embed.shape[0]), dtype=torch.bool)    # True => elements will NOT be processed
            #key_padding_mask = torch.cat((key_padding_mask_false, key_padding_mask_true), dim=0)


            # pad ength to max_length_t2s
            # add zeros at the end of text embed to reach max_length     
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
                text_embed = torch.cat((text_embed, pad), dim=0)

            # avg pooling
            #text_embed = torch.mean(text_embed, dim=0)

            # max pooling
            #text_embed = torch.amax(text_embed, dim=0)

            self.pointclouds.append({
                    'pointcloud': pc,
                    'text' : text,
                    'text_embed': text_embed,
                    'model_id': model_id,
                    'cate': category,
                    'mean': shift,
                    'std': scale,
                    #'key_pad_mask': key_padding_mask,
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['model_id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
        print(count_trunc ,' truncated text embeds')

    def get_ds_statistics(self, dataframe, from_shapenet_v1, from_shapenet_v2): # compute mean and std dev across required dataset
        dataframe_ = dataframe.drop_duplicates(subset=['modelId'])
        pointclouds=[]
        print('Applying global normalization...')
        print('Computing mean and std across all dataset...')
        count=0
        for idx, row in dataframe_.iterrows():
            count +=1
            # read_pcd
            model_id = row["modelId"]
            if from_shapenet_v1:
                model_path = str(self.root / "shapes" / "shapenet_v1" / f"{model_id}.ply")
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
                    
            elif from_shapenet_v2:
                model_path = str(self.root / "shapes" / "shapenet_v2" / f"{model_id}.ply")
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
            else:
                model_path = str(self.root / "shapes" / "text2shape" / f"{model_id}.ply")
                torch_cloud = IO().load_pointcloud(model_path)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc = torch.stack((points, colors), dim=1)
                    pc = pc.reshape(-1, 6)
                else:
                    pc = points
            pointclouds.append(pc)
        print('number of processed shapes: ', count)
        pcs = torch.cat(pointclouds, dim=0)
        mean = pcs.reshape(-1, 3).mean(axis=0).reshape(1,3)
        std = pcs.reshape(-1).std(axis=0).reshape(1,1)
        return mean, std

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else shallow_copy(v) for k, v in self.pointclouds[idx].items()}
        data["idx"] = idx
        if self.transform is not None:
            data = self.transform(data)
        
        return data

class Text2Shape_subset_mid(Text2Shape):
    def __init__(
        self,
        root: Path,
        chatgpt_prompts: bool,
        split: str,
        categories: str,
        from_shapenet_v1: bool,
        from_shapenet_v2: bool,
        language_model: str,
        lowercase_text: bool,
        max_length: int,
        padding: bool,
        conditional_setup: bool,
        scale_mode: str,
        chinese_distractor: bool = False,
        transforms: List[Callable] = []
    ) -> None:
    
        '''
        Class implementing a subset of Text2Shape dataset, where:
        - we provide a single entry for each model_id
        - the entry (cloud, text embedding, text...) is randomly sampled from the entries of that specific model_id
        - dataset size of training dataset with tables and chairs: 12054 entries
        
        '''

        super().__init__(root, chatgpt_prompts, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, 
                        max_length, padding, conditional_setup=True, scale_mode=scale_mode, transforms=transforms) # initialize parent Text2Shape

        # get unique model_ids
        m_ids = [shape["model_id"] for shape in self.pointclouds]
        self.m_ids = list(set(m_ids))

    
    def __getitem__(self, idx): 
        m_id = self.m_ids[idx]
        # find number of occurences of the current shape
        count=0
        for cloud in self.pointclouds:
            if cloud["model_id"]==m_id:
                count+=1 
        
        # pick a random index
        rand_idx = random.randrange(0, count)

        # return the corresponding entry of the dataset
        count=0
        for cloud in self.pointclouds:
            if cloud["model_id"]==m_id:
                if count==rand_idx:
                    cloud["idx"] = idx
                    return cloud
                else:
                    count+=1 

    def __len__(self):
        return len(self.m_ids)

class Text2Shape_pairs(Text2Shape):
    def __init__(
        self,
        root: Path,
        chatgpt_prompts: bool,
        split: str,
        categories: str,
        from_shapenet_v1: bool,
        from_shapenet_v2: bool,
        language_model: str,
        lowercase_text: bool,
        max_length: int,
        padding: bool,
        conditional_setup: bool,
        scale_mode: str,
        shape_gt: str,
        shape_dist: str,
        same_class_pairs: bool, # if True, each pair will have shapes from the same class (chair-chair or table-table)
        transforms: List[Callable] = []
    ) -> None:
    
        super().__init__(root, chatgpt_prompts, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, 
                        max_length, padding, conditional_setup, scale_mode, transforms) # initialize parent Text2Shape
        
        self.max_len = max_length
        self.shape_gt = shape_gt
        self.shape_dist = shape_dist
        self.same_class_pairs = same_class_pairs


    def __getitem__(self, idx): # build pairs of clouds
        target_mid = self.pointclouds[idx]["model_id"]
        target_cate = self.pointclouds[idx]["cate"]
        target_idx = idx
        dist_idx = idx
        dist_mid = target_mid

        if self.shape_gt is None:               # when shape_1 and shape_2 are None => we build pairs of GT T2S and random T2S
            assert self.shape_dist is None
            while dist_mid == target_mid or (self.same_class_pairs and target_cate!=self.pointclouds[dist_idx]["cate"]):       # we randomly sample a shape which is different from the current one, but belonging to the same class
                dist_idx = random.randint(0, len(self.pointclouds)-1)
                dist_mid = self.pointclouds[dist_idx]["model_id"]
            
            # build pair
            idxs = [target_idx, dist_idx]
            target = 0
            idxs, target = shuffle_ids(idxs, target)    # shuffle ids
            clouds = torch.stack((self.pointclouds[idxs[0]]["pointcloud"], self.pointclouds[idxs[1]]["pointcloud"]))
            cates = [self.pointclouds[idxs[0]]["cate"], self.pointclouds[idxs[1]]["cate"]]
            mids = [self.pointclouds[idxs[0]]["model_id"], self.pointclouds[idxs[1]]["model_id"]]

            # replace name of objects with corresponding indices of ShapeNetPart
            class_labels = []
            for cate in cates:
                if cate=="Chair":
                    class_labels.append(4)
                elif cate=="Table":
                    class_labels.append(15)

            class_labels = torch.Tensor(class_labels)

        else:
            # build pair with GT and DIST shapes
            target_clouds =  np.load(os.path.join(self.shape_gt, 'out.npy'))
            target_clouds = torch.from_numpy(target_clouds)
            target_cloud = target_clouds[target_idx]

            dist_clouds =  np.load(os.path.join(self.shape_dist, 'out.npy'))
            dist_clouds = torch.from_numpy(dist_clouds)
            dist_cloud = dist_clouds[target_idx]

            clouds = [target_cloud, dist_cloud]

            target=0
            clouds, target = shuffle_ids(clouds, target)

            # if one shape has only 2025 points => sample 2025 points from the other cloud.
            # TODO: you can also sample 2048 pts from 2025 pts => investigate this...
            #if clouds[0].shape[0] > clouds[1].shape[0]:
            #    clouds[0] = farthest_point_sampling(clouds[0], clouds[1].shape[0])
            #elif clouds[0].shape[0] < clouds[1].shape[0]:
            #    clouds[1] = farthest_point_sampling(clouds[1], clouds[0].shape[0])

            clouds = torch.stack((clouds[0], clouds[1]))

            class_labels = torch.Tensor()
            cates = None

        mean_text_embed = self.pointclouds[target_idx]["text_embed"]
        
        # compute mean, ignoring zeros of padding
        sum_embed = torch.sum(mean_text_embed, dim=1)

        seq_len = torch.count_nonzero(sum_embed)
        mean_text_embed = mean_text_embed[:seq_len,:]
        
        mean_text_embed = torch.mean(mean_text_embed, dim=0)
        
        text = self.pointclouds[target_idx]["text"]

        data = {"clouds": clouds,
                "mids": mids,
                "cates": cates,
                "target": target,
                "mean_text_embed": mean_text_embed,
                "text_embed": self.pointclouds[target_idx]["text_embed"],
                "text": text,
                "class_labels": class_labels,                              
                "idx": target_idx}
        
        #visualize_data_sample(clouds, target, str(text + f'_target={target}'), f"data_{self.split}_{idx}_{cates}.png", target_idx)   # RED:target, BLUE:distractor

        return data

class Text2Shape_pairs_easy_hard(Dataset):
    def __init__(
            self,
            device: torch.device,
            lang_model,
            lang_tokenizer,
            root: Path,
            chatgpt_prompts: bool,
            split: str,
            categories: str,
            from_shapenet_v1: bool,
            from_shapenet_v2: bool,
            language_model: str,
            lowercase_text: str,
            max_length: int,
            padding: bool,
            scale_mode: str,
            transforms: List[Callable] = [],
            shorten = False  # for DEBUGGING
    ) -> None:
        
        '''
        
        Class implementing Dataset with pairs of shape (easy or hard), with text description.

        '''

        super().__init__()
        print('Initializing dataset...')
        self.root = root
        self.split = split
        self.categories = categories
        self.scale_mode = scale_mode
        self.lowercase = lowercase_text
        self.transform = Compose(transforms)
        self.chatgpt_prompts = chatgpt_prompts

        if self.chatgpt_prompts:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"1e2h_{self.split}_{self.categories}_gpt2s.csv"
        else:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"1e2h_{self.split}_{self.categories}_t2s.csv"

        print('annotations path: ', annotations_path)
        df = pd.read_csv(annotations_path, quotechar='"', on_bad_lines='warn')   # cols: gt_id, dist_id, task, embed_dist, cate_gt, text, tensor
        
        count_trunc = 0
        self.data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="loading dataset"):
            
            if shorten != False and idx==shorten:
                break
            
            gt_id = row["gt_id"]
            dist_id = row["dist_id"]
            task = row["task"]
            embed_dist = row["embed_dist"]
            cate = row["cate_gt"]
            text = row["text"]
            tensor_name = row["tensor"]

            # build pair of clouds
            mids = [gt_id, dist_id]
            target = 0
            mids, target = shuffle_ids(mids, target)
            # pick clouds by model_id, from desired location
            if from_shapenet_v1:
                model_path_0 = str(self.root / "shapes" / "shapenet_v1" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                    
                model_path_1 = str(self.root / "shapes" / "shapenet_v1" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
                    
            elif from_shapenet_v2:
                model_path_0 = str(self.root / "shapes" / "shapenet_v2" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                    
                model_path_1 = str(self.root / "shapes" / "shapenet_v2" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
            else:
                model_path_0 = str(self.root / "shapes" / "text2shape_rgb" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                model_path_1 = str(self.root / "shapes" / "text2shape_rgb" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
                        
            # normalize clouds
            if self.scale_mode == 'shape_unit':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1)                    
            elif self.scale_mode == 'shape_half':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.5)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.75)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = (pc_max_0 - pc_min_0).max().reshape(1, 1) / 2
                    pc_max_1, _ = pc_1[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_1, _ = pc_1[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_1 = ((pc_min_1 + pc_max_1) / 2).view(1, 3)
                    scale_1 = (pc_max_1 - pc_min_1).max().reshape(1, 1) / 2
            elif self.scale_mode == 'shapenet_v1_norm':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = torch.linalg.norm(pc_max_0 - pc_min_0).reshape(1, 1)
                    pc_max_1, _ = pc_1[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_1, _ = pc_1[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_1 = ((pc_min_1 + pc_max_1) / 2).view(1, 3)
                    scale_1 = torch.linalg.norm(pc_max_1 - pc_min_1).reshape(1, 1)            
            else:
                    shift_0 = torch.zeros([1, 3])
                    scale_0 = torch.ones([1, 1])
                    shift_1 = torch.zeros([1, 3])
                    scale_1 = torch.ones([1, 1])
                    
            pc_0[:,:3] = (pc_0[:,:3] - shift_0) / scale_0
            pc_1[:,:3] = (pc_1[:,:3] - shift_1) / scale_1
            
            clouds = torch.stack((pc_0, pc_1))
        
            # build text embedding
            if self.lowercase:
                if padding:
                    if chatgpt_prompts==True:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                    else:
                        text_embed_path = self.root / "text_embeds" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts==True:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                if padding:
                    if chatgpt_prompts==True:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" /  "padding" / f"{tensor_name}"
                    text_embed_path = self.root / "text_embeds" / language_model / "std" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts==True:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "std" / f"{tensor_name}"
        
            if padding:
                text_embed = torch.load(text_embed_path, map_location=device)
            else:
                text_embed = torch.load(text_embed_path, map_location=device)
                '''
                # compute text embedding manually
                text = text.lower()
                input_text = lang_tokenizer(text, return_tensors="pt")
                input_text = input_text.to(device)
                # compute text embedding
                output = lang_model(input_ids = input_text.input_ids, attention_mask = input_text.attention_mask, output_hidden_states=True)
                last_hidden_state = output.last_hidden_state.detach()   # torch.Size([1, <seq length>, 1024])
                text_embed = last_hidden_state.squeeze(0)
                '''
                
                if text_embed.shape[0] > max_length:
                    count_trunc += 1
                    text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
                
            # build mask for this text embedding (i have text embeds length and max length)
            #key_padding_mask_false = torch.zeros((1+text_embed.shape[0]), dtype=torch.bool)             # False => elements will be processed
            #key_padding_mask_true = torch.ones((max_length - text_embed.shape[0]), dtype=torch.bool)    # True => elements will NOT be processed
            #key_padding_mask = torch.cat((key_padding_mask_false, key_padding_mask_true), dim=0)


            # pad length to max_length_t2s
            # add zeros at the end of text embed to reach max_length     
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
                pad = pad.to(device)
                text_embed = torch.cat((text_embed, pad), dim=0)


            # define text
            text = row["text"]

            clouds = clouds.detach().cpu()
            text_embed = text_embed.detach().cpu()

            self.data.append({
                    "clouds": clouds,
                    "mids": mids,
                    "target": target,
                    "text_embed": text_embed,
                    "text": text,
                    "task": task,
                    "embed_dist": embed_dist,
                    "cate": cate
                    })

            #visualize_data_sample(clouds, target, str(text + f'_target={target}'), f"data_2_{idx}_.png", idx)   # RED:target, BLUE:distractor
            
        # Deterministically shuffle the dataset        
        random.Random(2023).shuffle(self.data)
        print(count_trunc ,' truncated text embeds')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else shallow_copy(v) for k, v in self.data[idx].items()}
        data["idx"] = idx
        if self.transform is not None:
            data = self.transform(data)
        
        return data


class Text2Shape_unique_shapes_in_batch(Dataset):
    def __init__(
            self,
            device: torch.device,
            lang_model,
            lang_tokenizer,
            root: Path,
            chatgpt_prompts: bool,
            split: str,
            categories: str,
            from_shapenet_v1: bool,
            from_shapenet_v2: bool,
            language_model: str,
            lowercase_text: str,
            max_length: int,
            padding: bool,
            scale_mode: str,
            batch_size: int,
            shuffle_every: int,
            csv_alt_root: Path = '',        # to be removed later
            transforms: List[Callable] = [],
            ignore_short_batch = False,  # whether to drop the last batch if it is shorter than the others
            shorten = False,
    ) -> None:
        '''
        To be used with a pre-batched csv dataset. Elements are shuffled only inside each batch.
        Use RandomBatchOrderSampler as a sampler.
        '''

        super().__init__()
        print('Initializing dataset...')
        self.root = root
        self.categories = categories
        self.scale_mode = scale_mode
        self.lowercase = lowercase_text
        self.transform = Compose(transforms)
        self.chatgpt_prompts = chatgpt_prompts
        self.bsize = batch_size
        #split is "train", "val" or "test"

        #assert categories=='all'
        
        if self.chatgpt_prompts:
            if split=='train':
                annotations_path = (csv_alt_root if csv_alt_root else root) / f"train_{categories}_clean_batched.csv"
            else:
                annotations_path = (csv_alt_root if csv_alt_root else root) / f"val_{categories}_clean_batched.csv"
        else:
            raise NotImplementedError

        print(' - annotations path: ', annotations_path, '-')
        df = pd.read_csv(annotations_path, quotechar='"', on_bad_lines='warn')  
            # cols: gt_id, dist_id, task, embed_dist, cate_gt, text, tensor
        
        count_trunc = 0
        self.data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="loading dataset"):
            
            if shorten!=False and idx==shorten:
                break  # DEBUG
            
            id = row["modelId"]
            #task = row["task"]
            #embed_dist = row["embed_dist"]
            cate = row["category"]
            text = row["description"]
            tensor_name = row['tensor']

            # pick clouds by model_id, from desired location
            if from_shapenet_v1 or from_shapenet_v2:
                raise NotImplementedError
            else:
                model_path = str(self.root / "shapes" / "text2shape_rgb" / f"{id}.ply")
                torch_cloud = IO().load_pointcloud(model_path, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    raise NotImplementedError
                        
            # normalize cloud
            if self.scale_mode == 'shape_unit':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1)                   
            elif self.scale_mode == 'shape_half':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = (pc_max_0 - pc_min_0).max().reshape(1, 1) / 2
            elif self.scale_mode == 'shapenet_v1_norm':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = torch.linalg.norm(pc_max_0 - pc_min_0).reshape(1, 1)            
            else:
                    shift_0 = torch.zeros([1, 3])
                    scale_0 = torch.ones([1, 1])
                    
            pc_0[:,:3] = (pc_0[:,:3] - shift_0) / scale_0
            
            #clouds = torch.stack((pc_0, pc_1))
        
            if padding or (not chatgpt_prompts) or (not self.lowercase):
                raise NotImplementedError
            
            # build text embedding
            text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
        
            text_embed = torch.load(text_embed_path, map_location=device)
            if text_embed.shape[0] > max_length:
                count_trunc += 1
                text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
            # pad length to max_length_t2s
            # add zeros at the end of text embed to reach max_length     
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
                pad = pad.to(device)
                text_embed = torch.cat((text_embed, pad), dim=0)

            text_embed = text_embed.detach().cpu()

            self.data.append({
                    "clouds": pc_0.detach().cpu(),  # ACTUALLY A SINGLE CLOUD, name kept for compatibility
                    #"mids": mids,
                    #"target": target,
                    "text_embed": text_embed,
                    "text": text,
                    #"task": task,
                    #"embed_dist": embed_dist,
                    "cate": cate,
                    "original_id": id
                    })
            
            # Manual shuffle INSIDE BATCH ONLY:
            if idx and (idx%shuffle_every == 0):
                dcopy = self.data[idx-shuffle_every:idx]
                random.Random(int(time.monotonic())).shuffle(dcopy)
                self.data[idx-shuffle_every:idx] = dcopy
                # LAST batch not shuffled now!
                
            #visualize_data_sample(clouds, target, str(text + f'_target={target}'), f"data_2_{idx}_.png", idx)   # RED:target, BLUE:distractor
        
        k = len(df)//self.bsize*self.bsize
        if  ignore_short_batch and (len(df)-k)>0:
               self.data = self.data[:k]
               print(len(df) - k, 'discarded')
        
        print(count_trunc ,' truncated text embeds')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else shallow_copy(v) for k, v in self.data[idx].items()}
        data["idx"] = idx
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    

class Text2Shape_unique_shapes_in_batch_2(): # Dataset
    '''
    This version starts from the original CSV. Shuffling is performed before batching on the whole
    dataset, not inside batch.
    
    At first, a dictionary is created: {shape_id: shuffled_list_of_samples}
    In order to return unique shapes inside each batch, this object is implemented as an iterator; at
    each iteration, batch_size shapes are sampled from the dictionary keys and for each sampled shape,
    an element is popped from the correspondent list. When a list becomes empty, the key is removed from
    the dictionary.
    When there are not enough keys to form a batch, StopIteration is raised. This will discard some
    samples, but not always the same ones, due to non-deterministic sampling.
    '''
    def __init__(
            self,
            device: torch.device,
            lang_model,
            lang_tokenizer,
            root: Path,
            chatgpt_prompts: bool,
            split: str,
            categories: str,
            from_shapenet_v1: bool,
            from_shapenet_v2: bool,
            language_model: str,
            lowercase_text: str,
            max_length: int,
            padding: bool,
            scale_mode: str,
            batch_size: int,
            csv_alt_root: Path = '',        # to be removed later
            transforms: List[Callable] = [],
            shorten = False,
    ) -> None:
        #TODO description
        #super().__init__()
        print('Initializing dataset...')
        self.root = root
        self.categories = categories
        self.scale_mode = scale_mode
        self.lowercase = lowercase_text
        self.transform = Compose(transforms)
        self.chatgpt_prompts = chatgpt_prompts
        self.bsize = batch_size
        self.per_shape_DS = defaultdict(list)  # NEW!
        
        if self.chatgpt_prompts:
            if split=='train':
                annotations_path = (csv_alt_root if csv_alt_root else root) / f"train_{categories}_clean_batched.csv"
            else:
                annotations_path = (csv_alt_root if csv_alt_root else root) / f"val_{categories}_clean_batched.csv"
        else:
            raise NotImplementedError

        print(' - annotations path: ', annotations_path, '-')
        df = pd.read_csv(annotations_path, quotechar='"', on_bad_lines='warn')  
            # cols: gt_id, dist_id, task, embed_dist, cate_gt, text, tensor
        
        count_trunc = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="loading dataset"):
            if shorten and idx==shorten:
                break
            id = row["modelId"]
            #task = row["task"]
            #embed_dist = row["embed_dist"]
            cate = row["category"]
            text = row["description"]
            tensor_name = row['tensor']
            
            # pick clouds by model_id, from desired location
            if from_shapenet_v1 or from_shapenet_v2:
                raise NotImplementedError
            else:
                model_path = str(self.root / "shapes" / "text2shape_rgb" / f"{id}.ply")
                torch_cloud = IO().load_pointcloud(model_path, device=device)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    raise NotImplementedError
                        
            # normalize cloud
            if self.scale_mode == 'shape_unit':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1)                   
            elif self.scale_mode == 'shape_half':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = (pc_max_0 - pc_min_0).max().reshape(1, 1) / 2
            elif self.scale_mode == 'shapenet_v1_norm':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = torch.linalg.norm(pc_max_0 - pc_min_0).reshape(1, 1)            
            else:
                    shift_0 = torch.zeros([1, 3])
                    scale_0 = torch.ones([1, 1])
                    
            pc_0[:,:3] = (pc_0[:,:3] - shift_0) / scale_0
            
            #clouds = torch.stack((pc_0, pc_1))
        
            if padding or (not chatgpt_prompts) or (not self.lowercase):
                raise NotImplementedError
            
            # build text embedding
            text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
        
            text_embed = torch.load(text_embed_path, map_location=device)
            if text_embed.shape[0] > max_length:
                count_trunc += 1
                text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
            # pad length to max_length_t2s
            # add zeros at the end of text embed to reach max_length
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
                pad = pad.to(device)
                text_embed = torch.cat((text_embed, pad), dim=0)

            text_embed = text_embed.detach().cpu()
            
            self.per_shape_DS[id].append({
                "clouds": pc_0.detach().cpu(),  # ACTUALLY A SINGLE CLOUD, name kept for compatibility
                "text_embed": text_embed,
                "text": text,
                "cate": cate,
                "original_id": id
            })
        self.last_idx = idx
        
        self.per_shape_DS_orig = deepcopy(self.per_shape_DS)
        
        #k = len(df)//self.bsize*self.bsize
        #if  ignore_short_batch and (len(df)-k)>0:
        #       self.data = self.data[:k]
        #       print(len(df) - k, 'discarded')
        
        print(count_trunc ,' truncated text embeds')
    
    def __prepare__(self):
        self.per_shape_DS = deepcopy(self.per_shape_DS_orig)
        for id_, l in self.per_shape_DS.items():
            random.shuffle(l)
            #self.per_shape_DS[id_] = [len(l), l]
    
    def __next__(self):
        ''' Returns one entire batch! '''
        # finché N >= batch_size
            # estrai batch_size shape a caso tra le N disponibili
            # estrai un elemento della lista/pop per ciascuna; se poi la lista è vuota rimuovi la chiave.
            # ritorna le shape estratte
        # check discarded shapes and descr.
        # rebuild
        
        N = len(self.per_shape_DS)
        if N >= self.bsize:
            batch = []
            selected = random.sample(list(self.per_shape_DS), self.bsize)
            for shape in selected:
                data_list = self.per_shape_DS[shape]
                e = data_list.pop()
                if self.transform is not None: # TODO: bring outside
                    e = self.transform(e)
                batch.append(e)
                if len(data_list)==0:
                    del self.per_shape_DS[shape]
            #self.last_idx -= self.bsize
            return torch.utils.data.default_collate(batch)
        else:
            #print('stopping', N, self.bsize)
            raise StopIteration
        #N = len(self.per_shape_DS)
        #data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.data[idx].items()}
        #data["idx"] = idx
        #if self.transform is not None:
        #    data = self.transform(data)
        
    def __len__(self):
        return (self.last_idx+1) // self.bsize
    
    def __iter__(self):
        #N = len(self.per_shape_DS)
        #if N >= self.bsize:
        self.__prepare__()
        return self

class Text2Shape_humaneval(Dataset):
    '''
    dataset with T2S pointclouds where text and pairs are taken from JSON file of Human Evaluation
    '''
    
    def __init__(
        self,
        device: torch.device,
        lang_model,
        lang_tokenizer,
        json_path: Path,
        t2s_root: Path,
        categories: str,
        from_shapenet_v1: bool,
        from_shapenet_v2: bool,
        language_model: str,
        lowercase_text: bool,
        max_length: int,
        padding: bool,
        scale_mode: str,
        transforms: List[Callable] = [],
        restrict_to: List[str] = ['gpt2s', 't2s'],
        ) -> None:
        
        super().__init__()
        print(f'Building Dataset from JSON{" (RESTRICTED!)" if len(restrict_to)<2 else ""}...')
        
        self.t2s_root = t2s_root
        self.scale_mode = scale_mode
        self.transform = Compose(transforms)
        self.categories = categories
        
        # read JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        if categories != "all":
            # build list of model_ids with this shape
            cate_mids = []
            test_ids = Text2Shape(
                        root = self.t2s_root,
                        chatgpt_prompts = True,
                        split = 'test',
                        categories = self.categories,
                        from_shapenet_v1 = False,
                        from_shapenet_v2 = False,
                        language_model = 't5-11b',
                        lowercase_text = True,
                        max_length = 77,
                        padding = False,
                        conditional_setup = False,
                        scale_mode = "shapenet_v1_norm")
    
            for i in range(len(test_ids)):
                cate_mids.append(test_ids[i]["model_id"])
        
        self.data = []
        count_trunc = 0
        for idx, d in enumerate(tqdm(json_data, desc='Parsing JSON data...')):
            gt_id = d["gt_id"]
            dist_id = d["dist_id"]
            if categories!= "all" and (gt_id not in cate_mids or dist_id not in cate_mids):
                continue
            text = d["text"]
            dataset = d["dataset"]
            if dataset not in restrict_to:
                continue
            tensor_name = d["tensor_name"]
            task = d["task"]
            
            # build pair of clouds
            mids = [gt_id, dist_id]
            target = 0
            mids, target = shuffle_ids(mids, target)
            # pick clouds by model_id, from desired location
            if from_shapenet_v1:
                model_path_0 = str(self.t2s_root / "shapes" / "shapenet_v1" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                
                model_path_1 = str(self.t2s_root / "shapes" / "shapenet_v1" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
                    
            elif from_shapenet_v2:
                model_path_0 = str(self.t2s_root / "shapes" / "shapenet_v2" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                    
                model_path_1 = str(self.t2s_root / "shapes" / "shapenet_v2" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
            else:
                model_path_0 = str(self.t2s_root / "shapes" / "text2shape_rgb" / f"{mids[0]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_0)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_0 = torch.stack((points, colors), dim=1)
                    pc_0 = pc_0.reshape(-1, 6)
                else:
                    pc_0 = points
                    
                model_path_1 = str(self.t2s_root / "shapes" / "text2shape_rgb" / f"{mids[1]}.ply")
                torch_cloud = IO().load_pointcloud(model_path_1)
                points = torch_cloud.points_list()[0]
                if torch_cloud.features_list() is not None:
                    colors = torch_cloud.features_list()[0]
                    pc_1 = torch.stack((points, colors), dim=1)
                    pc_1 = pc_1.reshape(-1, 6)
                else:
                    pc_1 = points
                        
            # normalize clouds
            if self.scale_mode == 'shape_unit':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1)                    
            elif self.scale_mode == 'shape_half':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.5)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift_0 = pc_0[:,:3].mean(dim=0).reshape(1, 3)
                    scale_0 = pc_0[:,:3].flatten().std().reshape(1, 1) / (0.75)
                    shift_1 = pc_1[:,:3].mean(dim=0).reshape(1, 3)
                    scale_1 = pc_1[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = (pc_max_0 - pc_min_0).max().reshape(1, 1) / 2
                    pc_max_1, _ = pc_1[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_1, _ = pc_1[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_1 = ((pc_min_1 + pc_max_1) / 2).view(1, 3)
                    scale_1 = (pc_max_1 - pc_min_1).max().reshape(1, 1) / 2
            elif self.scale_mode == 'shapenet_v1_norm':
                    pc_max_0, _ = pc_0[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_0, _ = pc_0[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_0 = ((pc_min_0 + pc_max_0) / 2).view(1, 3)
                    scale_0 = torch.linalg.norm(pc_max_0 - pc_min_0).reshape(1, 1)
                    pc_max_1, _ = pc_1[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min_1, _ = pc_1[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift_1 = ((pc_min_1 + pc_max_1) / 2).view(1, 3)
                    scale_1 = torch.linalg.norm(pc_max_1 - pc_min_1).reshape(1, 1)            
            else:
                    shift_0 = torch.zeros([1, 3])
                    scale_0 = torch.ones([1, 1])
                    shift_1 = torch.zeros([1, 3])
                    scale_1 = torch.ones([1, 1])
                    
            pc_0[:,:3] = (pc_0[:,:3] - shift_0) / scale_0
            pc_1[:,:3] = (pc_1[:,:3] - shift_1) / scale_1
            
            clouds = torch.stack((pc_0, pc_1))
            
            # build text embedding
            if lowercase_text:
                if padding:
                    if dataset=="gpt2s":
                        text_embed_path = self.t2s_root / "text_embeds_chatgpt" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                    else:
                        text_embed_path = self.t2s_root / "text_embeds" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                elif dataset=="gpt2s":
                    text_embed_path = self.t2s_root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
                else:
                    text_embed_path = self.t2s_root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                if padding:
                    if dataset=="gpt2s":
                        text_embed_path = self.t2s_root / "text_embeds_chatgpt" / language_model / "std" /  "padding" / f"{tensor_name}"
                    text_embed_path = self.t2s_root / "text_embeds" / language_model / "std" /  "padding" / f"{tensor_name}"
                elif dataset=="gpt2s":
                    text_embed_path = self.t2s_root / "text_embeds_chatgpt" / language_model / "std" / f"{tensor_name}"
                else:
                    text_embed_path = self.t2s_root / "text_embeds" / language_model / "std" / f"{tensor_name}"
            
            
            if padding:
                text_embed = torch.load(text_embed_path, map_location='cpu')
            else:
                text_embed = torch.load(text_embed_path, map_location='cpu')
                '''
                # compute text embedding manually
                text = text.lower()
                input_text = lang_tokenizer(text, return_tensors="pt")
                input_text = input_text.to(device)
                # compute text embedding
                output = lang_model(input_ids = input_text.input_ids, attention_mask = input_text.attention_mask, output_hidden_states=True)
                last_hidden_state = output.last_hidden_state.detach()   # torch.Size([1, <seq length>, 1024])
                text_embed = last_hidden_state.squeeze(0)
                text_embed = text_embed.cpu()
                '''
                if text_embed.shape[0] > max_length:
                    count_trunc += 1
                    text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
                
            # build mask for this text embedding (i have text embeds length and max length)
            #key_padding_mask_false = torch.zeros((1+text_embed.shape[0]), dtype=torch.bool)             # False => elements will be processed
            #key_padding_mask_true = torch.ones((max_length - text_embed.shape[0]), dtype=torch.bool)    # True => elements will NOT be processed
            #key_padding_mask = torch.cat((key_padding_mask_false, key_padding_mask_true), dim=0)


            # pad length to max_length_t2s
            # add zeros at the end of text embed to reach max_length     
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
                text_embed = torch.cat((text_embed, pad), dim=0)

            
            mean_text_embed = torch.mean(text_embed, dim=0)        
        
            self.data.append({
                    "clouds": clouds,
                    "mids": mids,
                    "target": target,
                    "text_embed": text_embed,
                    "mean_text_embed": mean_text_embed,
                    "text": text,
                    "dataset": dataset,
                    "task": task
                    })
            pass
            
            #visualize_data_sample(clouds, target, str(text + f'_target={target}'), f"data_2__{idx}_.png", idx)   # RED:target, BLUE:distractor
            
        # Deterministically shuffle the dataset        
        random.Random(2023).shuffle(self.data)
        print(count_trunc ,' truncated text embeds')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else shallow_copy(v) for k, v in self.data[idx].items()}
        data["idx"] = idx
        if self.transform is not None:
            data = self.transform(data)
        
        return data
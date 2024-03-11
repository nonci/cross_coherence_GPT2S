import torch
import numpy as np 
import os.path as osp
import torch.nn.functional as F
import os
import random
import sys
from data.text2shape_dataset import Text2Shape, Text2Shape_pairs, Text2Shape_pairs_easy_hard, Text2Shape_humaneval, visualize_data_sample, visualize_data_sample_color
import json
import csv


from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from copy import copy
from collections import OrderedDict
import pandas as pd 
import matplotlib.pyplot as plt

from shapeglot.simple_utils import unpickle_data
from shapeglot.in_out.rnn_data_preprocessing import make_dataset_for_rnn_based_model
from shapeglot.in_out.shapeglot_dataset import ShapeglotDataset, T2SShapeglotDataset
from shapeglot.in_out.geometry import vgg_image_features, pc_ae_features
from shapeglot.models.neural_utils import MLPDecoder, smoothed_cross_entropy
from shapeglot.models.encoders import LanguageEncoder, PretrainedFeatures, PointCloudEncoder
from shapeglot.models.listener import Listener, T2S_Listener, Attention_Listener, Attention_Listener_Spatial
from shapeglot.vis_utils import visualize_example
from shapeglot import vis_utils

from train_autoencoder_color import ReconstructionNet
import argparse
from transformers import T5EncoderModel, T5Tokenizer

from my_config import get_config
config = get_config()

#DEF_CKPT = './exps/SIGMOID_loss_bt_TRAIN_1_all_bs16/checkpoint_30.pth'
#DEF_CKPT = './exps/SIGMOID_loss_simplif_1e-4_all/checkpoint_16.pth'
#DEF_CKPT = './exps/SIGMOID_loss_bt_TRAIN_3_all_bs16/checkpoint_13.pth'
DEF_CKPT = './exps/SIGMOID_bt=True_TRAIN_3_all_bs=16/checkpoint_6.pth'
LOSS = 'SIGMOID'
# LOSS and DEF_CKPT must be coherent: if you try to load a checkpoint from an
# architectures trained with another loss you should manually take care of b and t
# absence/presence!


def parse_args():

    # best chairs: 'attentionglot_gpt2s_vox_rgbclamp_1e2h_mean_expdecay_chair'
    # best overall: 'attentionglot_gpt2s_vox_rgbclamp_1e2h_mean_expdecay_all'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=DEF_CKPT, help='checkpoint path') # default='./exps/attentionglot_gpt2s_vox_rgbclamp_1e2h_mean_expdecay_all/checkpoint.pth'
    parser.add_argument('--dataset', type=str, default='humaneval', choices=['t2s', 'gpt2s', 'combined', 'humaneval'], help='dataset to use to build csv')
    parser.add_argument('--category', type=str, default='all', help='shape category to test on')
    parser.add_argument('--t2s_root', type=str, default=config.t2s_dataset, help='directory with T2S/GPT2S dataset')
    parser.add_argument('--json_path', type=str, default="./data/human_test_set_final.json", help='directory with T2S/GPT2S dataset')    
    parser.add_argument('--manual_text_embeds', action='store_true', help='if Set, text embeddings are computed again using LLM')
    parser.add_argument('--max_len', type=int, default=77, help="maximum length of text embeddings")
    parser.add_argument('--output_dir', type=str, default='./test/', help="set the output directory for the results")    
    
    opt = parser.parse_args()
    assert(LOSS in opt.ckpt)
    return opt

def main():
    # parse arguments
    args = parse_args()
    print(f'======= CHECKPOINT: {args.ckpt} =======')
    print(f'======= DATASET: {args.dataset} =======')
    print(f'======= CATEGORY: {args.category} =======')
    
    # create output directory if not present
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Directory '{args.output_dir}' created successfully.")
    else:
        print(f"Directory '{args.output_dir}' already exists.")

    # fix randomness to get always the same results, in multiple runs
    seed = 2023
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

    dataloaders     = {}
    datasets        = {}
    datasets_t2s    = {}
    datasets_gpt2s  = {}
    num_workers = 0
    batch_size = 100
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # overall accuracies (3 values for each experiment)
    accuracies =        {    "train": np.array([]),
                               "val": np.array([]),
                              "test": np.array([])}

    num_tests = 1
    wrong_idxs = []
    for split in ["test"]:
        if args.dataset=="combined":
            datasets_t2s[split] = Text2Shape_pairs(root=Path(args.t2s_root),
                                            chatgpt_prompts=False,
                                            split=split,
                                            categories=args.category,
                                            from_shapenet_v1=False,
                                            from_shapenet_v2=False,
                                            conditional_setup=True,
                                            language_model="t5-11b",
                                            lowercase_text=True,
                                            max_length=args.max_len,
                                            padding=False,
                                            scale_mode="shapenet_v1_norm",
                                            shape_gt=None,
                                            shape_dist=None,
                                            same_class_pairs=True)
            #np.random.seed(seed)
            #random.seed(seed)
            #torch.manual_seed(seed)
            #torch.cuda.manual_seed(seed)
            #torch.backends.cudnn.benchmark = False
            #torch.backends.cudnn.deterministic = True

            datasets_gpt2s[split] = Text2Shape_pairs(root=Path(args.t2s_root),
                                            chatgpt_prompts=True,
                                            split=split,
                                            categories=args.category,
                                            from_shapenet_v1=False,
                                            from_shapenet_v2=False,
                                            conditional_setup=True,
                                            language_model="t5-11b",
                                            lowercase_text=True,
                                            max_length=args.max_len,
                                            padding=False,
                                            scale_mode="shapenet_v1_norm",
                                            shape_gt=None,
                                            shape_dist=None,
                                            same_class_pairs=True)        

            datasets[split] = ConcatDataset([datasets_t2s[split], datasets_gpt2s[split]])

        elif args.dataset=='humaneval':
            if args.manual_text_embeds:
                lang_model_name = 't5-11b'
                t5_tokenizer = T5Tokenizer.from_pretrained(lang_model_name) #T5 Tokenizer is based on SentencePiece tokenizer: explanation here https://huggingface.co/docs/transformers/tokenizer_summary
                t5 = T5EncoderModel.from_pretrained(lang_model_name)
                t5 = t5.to(device)
            else:
                t5 = None
                t5_tokenizer = None
            datasets[split]=Text2Shape_humaneval(json_path=args.json_path,
                                                 device=device,
                                                 lang_model = t5,
                                                 lang_tokenizer = t5_tokenizer,
                                                t2s_root=Path(args.t2s_root),
                                                categories=args.category,
                                                #restrict_to = ['t2s'],
                                                from_shapenet_v1=False,
                                                from_shapenet_v2=False,
                                                language_model="t5-11b",
                                                lowercase_text=True,
                                                max_length=args.max_len,
                                                padding=False,
                                                scale_mode="shapenet_v1_norm")        
        else:
            if args.manual_text_embeds:
                lang_model_name = 't5-11b'
                t5_tokenizer = T5Tokenizer.from_pretrained(lang_model_name) #T5 Tokenizer is based on SentencePiece tokenizer: explanation here https://huggingface.co/docs/transformers/tokenizer_summary
                t5 = T5EncoderModel.from_pretrained(lang_model_name)
                t5 = t5.to(device)
            else:
                t5 = None
                t5_tokenizer = None
            datasets[split] = Text2Shape_pairs_easy_hard(
                                        device=device,
                                        lang_model=t5,
                                        lang_tokenizer=t5_tokenizer,
                                        root=Path(args.t2s_root),
                                        chatgpt_prompts=args.dataset=='gpt2s',
                                        split='val',
                                        categories=args.category,
                                        from_shapenet_v1=False,
                                        from_shapenet_v2=False,
                                        language_model="t5-11b",
                                        lowercase_text=True,
                                        max_length=args.max_len,
                                        padding=False,
                                        scale_mode="shapenet_v1_norm")
                    
        
        dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)

        del t5_tokenizer
        del t5
        torch.cuda.empty_cache()
        
        ## DEFINE MODELS
        shape_embedding_dim = 1024
        text_embedding_dim = 1024
        
        reconstr_net = ReconstructionNet(device, model_type='reconstruction_pointnet2', feat_dims=shape_embedding_dim, num_points=2048)
        pointnet_color = reconstr_net.encoder   # extract Encoder from ReconstructionNet
        
        feats_dim = 1024
        mlp_decoder = MLPDecoder(feats_dim, [100, 50, 1], use_b_norm=True, dropout=False) # originally, it is from 200 to 100, then 50, then 1. Here, it starts from embedding_dim 
        
        #listener = Attention_Listener_Spatial(pointnet_color, mlp_decoder, device, dim=640, n_heads=8, d_head=64, context_dim=embedding_dim, gated_ff=True, align=False, only_cross=True, no_residual=True, no_toout=True).to(device)
        #listener = Attention_Listener(pointnet_color, mlp_decoder, device, dim=640, n_heads=8, d_head=64, text_dim=text_embedding_dim).to(device)
        listener = Attention_Listener(
            config.clouds_dir,
            mlp_decoder,
            device,
            dim=640,
            n_heads=8,
            d_head=64,
            text_dim=text_embedding_dim,
            use_clip_loss=LOSS=='CLIP',
        ).to(device)

        ckpt = torch.load(args.ckpt, map_location=device)
        listener.load_state_dict(ckpt["model_state"])   
        print('loaded pre-trained model from ckpt at epoch ', ckpt["epoch"])
        
        for i in tqdm(range(num_tests), desc="Running experiments"):
            
            running_corrects = {"train":    torch.tensor(0),
                                "val":      torch.tensor(0),
                                "test":     torch.tensor(0)}

            acc = {}
            wrong_preds_idxs =   {"train":      torch.BoolTensor().to(device),
                                    "val":      torch.BoolTensor().to(device),
                                    "test":     torch.BoolTensor().to(device)}
            
            for phase in ["test"]:
                print('Dataset Size: ', len(datasets[phase]))
                count_wrong = {}
                results = []
                all_wrong = []  # wrong triplets, for debugging and final tests
                for idx, sample in enumerate(tqdm(dataloaders[phase], desc=f"{phase} set")):                    
                    
                    clouds = sample["clouds"].to(device)
                    mids = sample["mids"]
                    target = sample["target"].to(device)  

                    if 'dataset' in sample.keys():
                        dataset = sample["dataset"]
                    else:
                        dataset = [args.dataset] * len(mids[0])   # [gpt2s, gpt2s, ...] or [t2s, t2s, ...] 
                    
                    task = sample["task"]
                    texts = sample["text"]

                    #t5_tokenizer = T5Tokenizer.from_pretrained('t5-11b')
                    #tokens = [t5_tokenizer.tokenize(text) for text in texts]
                    
                    #with open('./visualization/tokens.csv', 'w', newline='') as file:
                    #    writer = csv.writer(file)
                    #    writer.writerows(tokens)

                    target_mids = []
                    dist_mids = []
                    for i, t in enumerate(target):
                        t = t.item()
                        target_mids.append(mids[t][i])
                        if t==0:
                            dist_mids.append(mids[1][i])
                        else:
                            dist_mids.append(mids[0][i])

                    text_embed = sample["text_embed"].to(device)
                    
                    #class_labels = sample["class_labels"]

                    listener.eval()

                    with torch.no_grad():
                        s1, sr = clouds.shape[0], clouds.shape[2:]   # assert clouds.shape[1]==2
                        mids = sample['mids']
                        ids = [mids[j][i] for i in range(0, len(mids[0])) for j in (0,1) ]  # mids[0] and mids[1] interleaved
                        logits = listener(
                            clouds.view(s1*2, *sr),
                            text_embed.repeat((2, *([1]*(len(text_embed.shape))))).permute(1,0,2,3).reshape(2*s1, *text_embed.shape[1:]),
                            ids=ids,
                        ).view(-1,2)
                        _, preds = torch.max(logits, 1)

                    # visualize first sample of the batch
                    visualize_data_sample_color(mids[0][target[0].item()], mids[0][1-target[0].item()], f'text: {texts[0]}\npred: {preds[0].item()}\ntarget: {target[0].item()}', f'datasample_{idx}.png', idx)

                    running_corrects[phase] = running_corrects[phase].to(device)
                    running_corrects[phase] += torch.sum(preds == target)
                    
                    for idx, tg in enumerate(target):
                        if preds[idx]==tg:
                            pred = 'gt'
                        else:
                            pred = 'dist'
                        if tg==0:
                            gt_id = mids[0][idx]
                            dist_id = mids[1][idx]
                        else:
                            gt_id = mids[1][idx]
                            dist_id = mids[0][idx]
                        
                        results.append({
                        'text':             texts[idx],
                        'gt_id':            gt_id,
                        'dist_id':          dist_id,
                        'pred':             pred})
                    
                    wrong_preds_idxs[phase] = torch.cat([wrong_preds_idxs[phase], preds!=target], dim=0)

                    if not torch.equal(preds, target):  # if some predictions are wrong
                        wrong_clouds = clouds[preds!=target]
                        indices = (preds!=target).nonzero().squeeze().tolist()
                        if isinstance(indices, list) and len(indices)>0:
                            wrong_target_mids = [target_mids[i] for i in indices]
                            wrong_dist_mids = [dist_mids[i] for i in indices]
                            wrong_dataset_task = [(dataset[i], task[i]) for i in indices]
                            for i in indices:
                                wrong_task = task[i]
                                wrong_dataset = dataset[i]
                                key = (wrong_task, wrong_dataset)
                                count_wrong[key] = count_wrong.get(key, 0) + 1
                        else:
                            wrong_target_mids = target_mids[indices]
                            wrong_dist_mids = dist_mids[indices]
                            wrong_dataset_task = (dataset[indices], task[indices])
                            wrong_task = task[indices]
                            wrong_dataset = dataset[indices]
                            key = (wrong_task, wrong_dataset)
                            count_wrong[key] = count_wrong.get(key, 0) + 1
                        #count_wrong += wrong_clouds.shape[0]
                        list_wrong = (preds!=target).tolist()
                        wrong_idxs = sample["idx"][list_wrong]
                        wrong_indices = [i for i, x in enumerate(list_wrong) if x == True]
                        wrong_texts = [texts[i] for i in wrong_indices]
                        all_wrong += list (zip([e for i,e in enumerate(sample['mids'][0]) if list_wrong[i]],
                                             [e for i,e in enumerate(sample['mids'][1]) if list_wrong[i]],
                                             [t_ for i,t_ in enumerate(sample['text']) if list_wrong[i]],
                                             [t.item() for i,t in enumerate(target) if list_wrong[i]],
                                             [l for i,l in enumerate(logits) if list_wrong[i]]
                                           ) )
                        ## Saving results to file
                        with open(os.path.join(args.output_dir, 'wrong_ids_attnglot.txt'), 'w') as f:
                            for i in range(wrong_idxs.shape[0]):
                                f.write(str(wrong_idxs[i].item()) + '\n')
                        f.close()    

                        with open(os.path.join(args.output_dir, 'mistakes.txt'), 'w') as f:
                            if isinstance(wrong_target_mids, list):
                                for i in range(len(wrong_target_mids)):
                                    f.write(f'target: {wrong_target_mids[i]}, dist: {wrong_dist_mids[i]}, dataset: {wrong_dataset_task[i]}\n')
                            else:
                                f.write(f'target: {wrong_target_mids}, dist: {wrong_dist_mids}, dataset: {wrong_dataset_task}\n')
                        f.close()    
                        
                        if wrong_clouds.shape[0]>1:
                            for i in range (len(wrong_clouds)):
                                img_title = f"{wrong_texts[i]} idx={wrong_idxs[i].item()}"
                                img_path = os.path.join(args.output_dir, f'sample_{wrong_target_mids[i]}_{wrong_dist_mids[i]}.png')
                                #visualize_data_sample(wrong_clouds[i], wrong_targets[i], img_title, img_path, i)
                                img_title = f"{wrong_texts[i]} idx={wrong_idxs[i].item()}"
                                img_path = os.path.join(args.output_dir, f'color_sample_{wrong_target_mids[i]}_{wrong_dist_mids[i]}.png')
                                #visualize_data_sample_color(wrong_target_mids[i], wrong_dist_mids[i], img_title, img_path, i)
                        else:
                                img_title = f"{wrong_texts} idx={wrong_idxs.item()}"
                                img_path = os.path.join(args.output_dir, f'color_sample_{wrong_target_mids}_{wrong_dist_mids}.png')
                                #visualize_data_sample_color(wrong_target_mids, wrong_dist_mids, img_title, img_path, i)                            

                n_examples = len(dataloaders[phase].dataset)
                
                acc[phase] = running_corrects[phase].double() / n_examples
                
                print(all_wrong[:10])
                
                accuracies[phase] = np.concatenate([accuracies[phase], np.expand_dims(acc[phase].cpu().numpy(), axis=0)], axis=0)

                print(f'{phase} accuracy: {acc[phase]}')
                #print('wrong predictions: ', count_wrong)
                print('correct predictions: ', running_corrects[phase])
                print('wrong predictions: ', count_wrong)
                print('all predictions: ', n_examples)

                # Extract the task and dataset combinations as labels and counts as values
                labels = [f'{task}-{dataset}' for task, dataset in count_wrong.keys()]
                values = list(count_wrong.values())

                # Create a bar plot
                plt.figure(figsize=(10, 5))
                plt.bar(labels, values)
                plt.xlabel('Task and Dataset')
                plt.ylabel('Occurrences')
                plt.title('Occurrences of Task and Dataset Combinations')
                plt.xticks(rotation=45)

                total = sum(values)
                for i, v in enumerate(values):
                    percentage = (v / total) * 100
                    plt.text(i, v + 1, f'{percentage:.1f}% ({v})', ha='center', color='red', weight='bold', fontsize=15)

                # Display the plot
                plt.show()
        
    
    # save results to file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        for phase in  ["test"]:
            f.write(f"{phase}: \n")
            for acc in accuracies[phase]:
                f.write(f"{acc}\t")
            f.write("\n")
    f.close
    
    with open('./results.json', "w") as f:
        json.dump(results, f, indent=4)

    print('Average accuracies:')
    for phase in ["test"]:
        mean_acc = np.mean(accuracies[phase])
        std_acc = np.std(accuracies[phase])
        print(f"{phase}: mean={mean_acc}, std={std_acc}")

if __name__ == '__main__':
    main()
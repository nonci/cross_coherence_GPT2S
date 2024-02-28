'''
Launch, for ex., with: --category all    #--json_path ./data/human_test_set_final.json
ADAPTED!
Compute CrossCoherence on HST groundtruth shapes, using the same paradygm of CLIP R-precision.
For every gt shape, I compute the CC between the shape and all the 153 texts (1 gt text and 152 distractive texts). 
I choose the highest logit as the prediction of the metric.
'''
VERBOSE = False

import argparse
import torch
import numpy as np
#import matplotlib.pyplot as plt
from test_attnglot_pointe_chinese_shap_e import build_mids_texts_tensornames, get_hst_cloud
#from pathlib import Path
from tqdm import tqdm
import random
import os
import json
from my_config import get_config
config = get_config()

# CrossCoherence submodules
from train_autoencoder_color import ReconstructionNet
from shapeglot.models.neural_utils import MLPDecoder
from shapeglot.models.listener import Attention_Listener_Spatial, Attention_Listener


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt', type=str, default='./exps/old/SIGMOID_loss_bt_TRAIN_3_all_bs16/checkpoint_13.pth', help='checkpoint path')
	parser.add_argument('--category', type=str, default="Chair", help='category of shapes to test')
	parser.add_argument('--json_path', type=str, default="./data/human_test_set_final.json", help='directory with HST dataset')    
	parser.add_argument('--seed', type=int, default=2023, help="seed used to fix randomness")
	parser.add_argument('--r', type=int, default=1, help="R value of R-precision")
	parser.add_argument('--t2s_root', type=str, default=config.t2s_dataset, help='directory with T2S/GPT2S dataset')
	parser.add_argument('--strict_check', action='store_true', help='if True, the prediction is correct if it matches the exact text which generated the shape')
	parser.add_argument('--n_dist', type=int, default=153, help='set the number of distractors for every text prompt')
	parser.add_argument('--all_dist_texts', action='store_true', help='if Set, we use all the text prompts from the dataset as distractors')
	
	opt = parser.parse_args()
	opt.category=opt.category.capitalize()
	print('CHECKPOINT', opt.ckpt)
	print('CAT', opt.category)
	return opt

def load_cc_model(ckpt_path, device):
	## DEFINE MODELS
	embedding_dim = 1024
	
	reconstr_net = ReconstructionNet(device, model_type='reconstruction_pointnet2', feat_dims=embedding_dim, num_points=2048)
	pointnet_color = reconstr_net.encoder   # extract Encoder from ReconstructionNet
	
	mlp_decoder = MLPDecoder(embedding_dim, [100, 50, 1], use_b_norm=True, use_dropout=False)
	# originally, it is from 200. Here, it starts from embedding_dim.

	listener = Attention_Listener(pointnet_color, mlp_decoder, device, dim=640, n_heads=8, d_head=64, context_dim=embedding_dim).to(device) # 640 when using color-aware PcEncoder, otherwise 512

	ckpt = torch.load(ckpt_path, map_location=device)
	listener.load_state_dict(ckpt["model_state"])
	print('loaded pre-trained model from ckpt at epoch ', ckpt["epoch"])
	
	return listener

########################################


def main():
	LOSS = 'SIGMOID'
	
	args = parse_args()
	device = "cuda:3" if torch.cuda.is_available else 'cpu'
	print(f'USING DEVICE {device}')
	embedding_dim = 1024
	
	mlp_decoder = MLPDecoder(embedding_dim, [100, 50, 1], use_b_norm=True, dropout=False) # originally, it is from 200 to 100, then 50, then 1. Here, it starts from embedding_dim 

	listener = Attention_Listener(
		config.clouds_dir,
		mlp_decoder,
		device,
		dim=640,
		n_heads=8,
		d_head=64,
		text_dim=embedding_dim,
		use_clip_loss=LOSS=='CLIP',
	).to(device)

	ckpt = torch.load(args.ckpt, map_location=device)
	listener.load_state_dict(ckpt["model_state"])
 
	num_exps = 1
	accuracies = []
	data_path = f'{config.t2s_dataset}/shapes/text2shape_rgb'
	
	for run_idx in range(0, num_exps):   
		mids_texts = build_mids_texts_tensornames(args.json_path, device)
		#all_texts = [d['text'] for d in mids_texts]

		miss_count = 0
		n_corrects = 0
		results_path = f'cc_r_precision_hst_{args.category}.txt' 
		all_mids_texts = mids_texts
		results = []
		count_samples = 0
		listener.eval()
		with open(results_path, 'a') as f:
			for idx, mid_text in enumerate(tqdm(all_mids_texts, desc="Computing CC R-precision")):
				gt_text = mid_text["text"]
				#gt_tensorname = mid_text["tensor_name"]
				#gt_dataset = mid_text["dataset"]
				gt_mid = mid_text["mid"]
				gt_cate = mid_text["cate"]
				
				if args.category != 'All' and gt_cate != args.category:
					continue
					
				count_samples += 1
				
				cloud, file_cloud = get_hst_cloud(gt_mid, data_path) # ret. normalized cloud and its file path

				cloud = cloud.unsqueeze(0).unsqueeze(0)  # to size [1, 1, 2048, 6]
				#visualize_pointcloud(cloud.squeeze(0).squeeze(0), save_fig=True, filename=f'hst_cloud_{idx}.png')

				logits_texts = []
				random.seed(idx)  # fix seed to make sure to get the same random texts at every run
				if not args.all_dist_texts:
					# All the texts of the same category:
					other_mids_texts = [d for d in all_mids_texts if d["text"] != gt_text and d["cate"] == gt_cate]
					# pick n-1 random elements from mids_texts:
					random_mids_texts = random.sample(other_mids_texts, args.n_dist-1)
					# Insert gt_text and gt_mid at run_idx position (so that we have gt at a known position):
					random_mids_texts.insert(run_idx, mid_text)
					if VERBOSE:
						print('\nall text prompts: ', len(random_mids_texts))
					
					for random_idx, random_mid_text in enumerate(random_mids_texts):
						random_text = random_mid_text["text"]
						random_tensorname = random_mid_text["tensor_name"]
						random_dataset = random_mid_text["dataset"]

						# get text embedding of random text
						if random_dataset == "gpt2s":
							text_embed_path = os.path.join(args.t2s_root, "text_embeds_chatgpt", "t5-11b", "lowercase", f"{random_tensorname}")
						else:
							text_embed_path = os.path.join(args.t2s_root, "text_embeds", "t5-11b", "lowercase", f"{random_tensorname}")
						
						text_embed = torch.load(text_embed_path, map_location=device)
						text_embed = text_embed.unsqueeze(0)  # to shape [1, 17, 1024]
						
						with torch.no_grad():
							# compute CC between shape and text
							logits = listener(cloud, text_embed, ids=[gt_mid])  # text=random_text
							# clouds:[1,2048,3] | text_embed:[1,seq,1024] => logits: [B,1]

							# store CC value (logit)
							logits_texts.append({'text': random_text,
												 'logit': logits.item()})

							if random_idx == run_idx:
								gt_logit = logits.item()
		

					# Sort logits
					sorted_dicts = sorted(logits_texts, key=lambda x: x['logit'], reverse=True)
					top_r_texts = [item['text'] for item in sorted_dicts[:args.r]]
					top_r_logits = [item['logit'] for item in sorted_dicts[:args.r]]
					top_r_values = sorted_dicts[:args.r]
					if VERBOSE:
						print('gt text: ', gt_text)
						print('predicted text: ', top_r_texts)
						  
					# get top R predictions
					# check if prediction is correct
					if gt_text in top_r_texts:
						n_corrects +=1
						if VERBOSE:
							print('CORRECT')
							print('correct samples:', n_corrects)
					elif VERBOSE:
						print('WRONG')

					if VERBOSE: 
						print(f'logit of gt_text: {gt_logit}')
						#print(f'ranking of gt_text: {gt_idx}')
						print('===========\n')
					
					results.append({'shape': file_cloud,
									'gt_text': gt_text,
									'gt_logit': gt_logit,
									'pred_text': top_r_values[0]["text"],
									'pred_logit': top_r_values[0]["logit"],
									'label': 'CORRECT' if gt_text in top_r_texts else 'WRONG',
									'rank_10': sorted_dicts[:10]})
							
			n_samples = count_samples - miss_count
			acc = n_corrects / n_samples
			accuracies.append(acc)
			print('miss count: ', miss_count)
			print('n_corrects: ', n_corrects)
			print('num samples: ', n_samples)
			print('Accuracy: ', acc)
			
			f.write(f'run: {run_idx}\n')
			f.write(f'n_corrects: {n_corrects}\n')
			f.write(f'samples: {n_samples}\n')
			f.write(f'accuracy: {acc}\n')
			f.write('======================================\n\n')
			f.close()
	
	mean_acc = np.mean(accuracies)
	std_acc = np.std(accuracies)
	
	with open(results_path, 'a') as f:
		f.write(f'mean: {mean_acc}\n')
		f.write(f'std: {std_acc}\n')
	f.close()
	
	with open(f'data/cc_r_precision_hst_{args.category}.json', 'w') as f_results:
		json.dump(results, f_results, indent=4)


if __name__ == "__main__":
	print('starting...')
	main()
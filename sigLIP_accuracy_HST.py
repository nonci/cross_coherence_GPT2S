# Test of SigLIP "Sigmoid loss for Language-Image Pre-training"
# adapted from https://huggingface.co/timm/ViT-B-16-SigLIP 

# This version works on GPU

#import PIL
import torch
from torchvision.transforms import *
import torch.nn.functional as F
import json, os, time
from PIL import Image
import inspect
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0,timm>=0.9.8
from my_config import get_config
config = get_config()

CATS = {'chair':3001627, 'table':4379243}

GPT2s_only = lambda _: _['dataset']=='gpt2s'
chairs_only = lambda _: (_['gt_cat']==CATS['chair']) and (_['dist_cat']==CATS['chair'])

##################################################################################################
TARGET_JSON = f'{config.base_dir}/data/human_test_set_final_with_cats.json'
RENDERS_FOLDER = config.render_dir
COMPUTE_TXT_FEAT = True
if not COMPUTE_TXT_FEAT:
	TXT_FEAT_FOLDER = config.text_emb_dir
FILTERS = [GPT2s_only]  # chairs_only, 
MODEL = 'hf-hub:timm/ViT-B-16-SigLIP'
FAST_PREPROC = False
#keys are: gt_id, dist_id, text, dataset, task, tensor_name, dist_cat, gt_cat
##################################################################################################

device = 'cuda:0'
assert(torch.cuda.is_available())

print(end='Loading model and tokenizer... ')
model, preprocess = create_model_from_pretrained(MODEL)  # sentence-transformers/clip-ViT-B-32  ?
model.to(device)
model.eval()
if FAST_PREPROC:
	preprocess.transforms = [
		ToTensor(),
		Lambda(lambda _: _.convert('RGB')),  # anticip. for efficiency -- "def _convert_to_rgb(image):\n    return image.convert('RGB')\n"
		Lambda(Lambda(lambda _: _.to(device))),
		Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
		Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
tokenizer = get_tokenizer(MODEL)
print('ok')

ok, bad, filter_discarded  = 0, 0, 0
start_t = time.time()

with open(TARGET_JSON, 'rb') as f:
    data = json.load(f)
    
if FILTERS:
    print('WARNING: using filter(s):', ', '.join([inspect.getsource(fil).partition('=')[0].strip() for fil in FILTERS]))

for sample in data:
	if not all(filter(sample) for filter in FILTERS):
		filter_discarded += 1
		continue

	if COMPUTE_TXT_FEAT:
		tk_texts = tokenizer(sample['text'], context_length=model.context_length).to(device) # [n_sentences, context_len]
		text_features = model.encode_text(tk_texts).detach() # [len(tk_texts), 768]
		text_features = F.normalize(text_features, dim=-1)
	else:
		text_features = torch.load(os.path.join(TXT_FEAT_FOLDER, sample['tensor_name']), map_location=device).detach()
	
	path_gt = os.path.join(RENDERS_FOLDER, f'0{sample["gt_cat"]}', sample['gt_id'])
	path_dist = os.path.join(RENDERS_FOLDER, f'0{sample["dist_cat"]}', sample['dist_id'])
	probs = {'dist': [], 'gt': []}
	text_probs = []  # temp. variable
	
	if not os.path.exists(path_gt):
		print(path_gt)
		continue
	if not os.path.exists(path_dist):
		print(path_dist)
		continue
 
	paths_gt = [os.path.join(path_gt, r) for r in os.listdir(path_gt)]
	paths_dist = [os.path.join(path_dist, r) for r in os.listdir(path_dist)]

	for path in paths_gt + paths_dist:
		with open(path, "rb") as imgf:
			#try:
			image = Image.open(imgf)
			#except PIL.UnidentifiedImageError:
			#	print('SKIP')
			#	continue
			image = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
		with torch.no_grad(), torch.cuda.amp.autocast():
			image_features = model.encode_image(image)  # [1, 768]
			image_features = F.normalize(image_features, dim=-1)
			text_probs.append(torch.sigmoid(
				image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
			).item()) # [1,len(tk_texts)]
	
	probs['gt'] = text_probs[:20]
	probs['dist'] = text_probs[20:]
	
	ok  += max(probs['gt']) >  max(probs['dist'])
	bad += max(probs['gt']) <= max(probs['dist'])

	print(end=f"\r{ok+bad:>6} - Acc. so far: {ok/(ok+bad):>.3f} ({round(time.time()-start_t)} s)", flush=True)

print('\n********** FINAL STATS ***********\n'
      f'OK  {ok:>7}\nBAD {bad:>7}\n'
      f'Accuracy: {ok/(ok+bad):>.5f}\n'
      f'Discarded due to filtering: {filter_discarded}\n'
      '***********************************'
)


'''
ORIG. PREPROC:

2153 - Acc. so far: 0.841 (1674 s)
********** FINAL STATS ***********
OK     1810
BAD     343
Accuracy: 0.84069
Discarded due to filtering: 0
*********************************

WARNING: using filter(s): chairs_only
   923 - Acc. so far: 0.841 (687 s)
********** FINAL STATS ***********
OK      776
BAD     147
Accuracy: 0.84074
Discarded due to filtering: 1230
*********************************

FAST PREPROC:
WARNING: using filter(s): chairs_only
   923 - Acc. so far: 0.839 (585 s)
********** FINAL STATS ***********
OK      774
BAD     149
Accuracy: 0.83857
Discarded due to filtering: 1230
*********************************

FAST PREPROC:
WARNING: using filter(s): chairs_only, GPT2s_only
   462 - Acc. so far: 0.853 (292 s)
********** FINAL STATS ***********
OK      394
BAD      68
Accuracy: 0.85281
Discarded due to filtering: 1691
*********************************
'''
#  chairs:  923 - Acc. so far: 0.841 (3059 s)
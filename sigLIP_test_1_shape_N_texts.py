# Test of SigLIP "Sigmoid loss for Language-Image Pre-training"
# adapted from https://huggingface.co/timm/ViT-B-16-SigLIP

from my_config import get_config
config = get_config()

##################################################################################################
CSVS = [
	'./data/test_all.csv', # chairs and tables
	'./data/test_all_gpt2s.csv', # chairs and tables
	'./data/train_all_clean.csv', # chairs and tables
]
RENDERS_FOLDER = config.render_dir2
ID = 'e6c900568268acf735836c728d324152'
RELATIVIZE_SCORES = True
##################################################################################################


import torch
import torch.nn.functional as F
import csv, os
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8


def get_texts_from_csv(fname, id_):
	'''Get text descriptions of the shape with id "id_" from one or more files.'''
	texts = []
	if type(fname) is str: fname=[fname]
	for fn in fname:
		with open(fn, "r") as f:
			reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			for row in reader:
				if row[1]!=id_: continue
				texts.append(row[2])
	return texts


model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
#labels_list = ["a dog", "a cat", "a donut", "chair"]
texts = get_texts_from_csv(CSVS[1], ID) + [
	"A beautiful pink armrest.",
	"A fat chicken."
]
tk_texts = tokenizer(texts, context_length=model.context_length) # [n_sentences, context_len]   was: model.context_length
extract_n = lambda _: int(_[_.find('_')+1:_.rfind('_')])
files = sorted(os.listdir(os.path.join(RENDERS_FOLDER, ID)), key=extract_n)
#files = files[:2]  # only some files (STOP)

for file in files:
	print(file)
	with open(os.path.join(RENDERS_FOLDER, ID, file), "rb") as imgf:
		image = Image.open(imgf)
		image = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
	
	with torch.no_grad(), torch.cuda.amp.autocast():
		image_features = model.encode_image(image)  # [1, 768]
		text_features = model.encode_text(tk_texts) # [len(tk_texts), 768]
		image_features = F.normalize(image_features, dim=-1)
		text_features = F.normalize(text_features, dim=-1) 

		text_probs = torch.sigmoid(
      		image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
        ) # [1,len(tk_texts)]
		#zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))

	if RELATIVIZE_SCORES:
		text_probs/=torch.max(text_probs)
		for i,t in enumerate(texts):
			print(f'   {round(text_probs[0][i].item(), 3):.2f} ', t)
	else:
		for i,t in enumerate(texts):
			print(f'   {text_probs[0][i].item(): .1e} ', t)	
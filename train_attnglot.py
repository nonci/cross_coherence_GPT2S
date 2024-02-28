from my_config import get_config
config = get_config()

##########################################
SHORTEN = 512 # False, 512,
BUILD_CACHE = False #True
batch_size = 32 # 2**6 max with SIGMOID  # Used for train and (if needed) test
val_batch_size = 2
shape_embedding_dim = 1024      # 2048 when using the color-aware PcEncoder with embed size 2048, 1024 when using the standard PcEncoder and std color-aware PcEncoder
text_embedding_dim = 1024       # 1024 when using T5 as text encoder   
train_epochs = 45
#reg_gamma = 0.001 #0.005       # Weight regularization on FC layers
dataroot = config.t2s_dataset
tmp_csv_root = config.t2s_dataset
splits = ["train", "val"]
categories = 'all'
manual_text_embeds = False
# Whether the CSV file to be used is the one with already batched data (in which shapes don't repeat) or not:
CSV_IS_BATCHED = False
LOSS = 'SIGMOID'
output_dir = f"./exps/ignore"
#output_dir = f"./exps/{LOSS}_loss_bt_TRAIN_3_{categories}_bs{batch_size}"  # Folder to save training data and checkpoints
##########################################

assert LOSS in ('SIGMOID', 'CLIP')
assert categories in ('all', 'chair', 'table')

if BUILD_CACHE:
    assert batch_size in (2,4,8)
    assert batch_size == val_batch_size
    assert not SHORTEN

import torch
import numpy as np
import os
import torch.nn.functional as F
import sys
from data.text2shape_dataset import Text2Shape, Text2Shape_humaneval, Text2Shape_unique_shapes_in_batch, Text2Shape_unique_shapes_in_batch_2
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
#from datetime import datetime
import argparse
import logging
from shapeglot.models.neural_utils import MLPDecoder
from shapeglot.models.listener import Attention_Listener, Attention_Listener_Spatial
from train_autoencoder_color import ReconstructionNet
from transformers import T5EncoderModel, T5Tokenizer
from custom_utils import  RandomBatchOrderSampler, permutation_indexes, visualize_pointcloud_pair #tensor_rotate, list_rotate

def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'logger.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    return logger

def main():
    # fix random seed to get always the same samples at each training experiment
    #random_seed = 2023
    #random.seed(random_seed)

    dataloaders = {}
    num_workers = 1
    max_length = 77         # max length of text embedding sequence, to truncate when loading the dataset
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print('CATEGORY:', categories)
    
    if manual_text_embeds:
        lang_model_name = 't5-11b'  # see: https://huggingface.co/t5-11b
        t5_tokenizer = T5Tokenizer.from_pretrained(lang_model_name) #T5 Tokenizer is based on SentencePiece tokenizer: explanation here https://huggingface.co/docs/transformers/tokenizer_summary
        t5 = T5EncoderModel.from_pretrained(lang_model_name)
        t5 = t5.to(device)
    else:
        t5 = None
        t5_tokenizer = None
    
    for split in splits:
        print('Loading DS:', split)
        
        if split=='train':
            if CSV_IS_BATCHED:
                # In this case we don't have to manually collate
                dataset = Text2Shape_unique_shapes_in_batch(
                    device=device,  lang_model=t5,
                    lang_tokenizer=t5_tokenizer,  root=Path(dataroot),
                    chatgpt_prompts=True,  split=split,
                    categories=categories,  from_shapenet_v1=False,
                    from_shapenet_v2=False,  language_model="t5-11b",
                    lowercase_text=True,  max_length=max_length,
                    padding=False,  scale_mode="shapenet_v1_norm",
                    shuffle_every=batch_size,  csv_alt_root=Path(tmp_csv_root),
                    batch_size=batch_size,  ignore_short_batch=True,
                    shorten=SHORTEN)
                sampler = RandomBatchOrderSampler(
                    len(dataset),
                    batch_size,
                    drop_last=True)
                dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                    num_workers=num_workers,
                    batch_sampler = sampler,)  # DROP LAST?
            else:
                dataloaders[split] = Text2Shape_unique_shapes_in_batch_2(
                    device=device,  lang_model=t5,
                    lang_tokenizer=t5_tokenizer,  root=Path(dataroot),
                    chatgpt_prompts=True,  split=split,
                    categories=categories,  from_shapenet_v1=False,
                    from_shapenet_v2=False,  language_model="t5-11b",
                    lowercase_text=True,  max_length=max_length,
                    padding=False,  scale_mode="shapenet_v1_norm",
                    csv_alt_root=Path(tmp_csv_root), batch_size=batch_size,
                    shorten=SHORTEN)
                # TEST:
                '''
                nb= 0 
                for b in dataloaders[split]:
                    assert len(set(b['clouds'])) == batch_size
                    nb+=1
                print('n batches',nb)
                '''
        else: #val
            if CSV_IS_BATCHED:
                dataset = Text2Shape_unique_shapes_in_batch(
                    device=device,  lang_model=t5,
                    lang_tokenizer=t5_tokenizer,  root=Path(dataroot),
                    chatgpt_prompts=True,  split=split,
                    categories=categories,  from_shapenet_v1=False,
                    from_shapenet_v2=False,  language_model="t5-11b",
                    lowercase_text=True,  max_length=max_length,
                    padding=False,  scale_mode="shapenet_v1_norm",
                    shuffle_every=val_batch_size, csv_alt_root=Path(tmp_csv_root),
                    batch_size=val_batch_size,  ignore_short_batch=True,
                    shorten=SHORTEN)  # DEBUG
                sampler = RandomBatchOrderSampler(
                    len(dataset),
                    val_batch_size,
                    drop_last=True)
                dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                    num_workers=num_workers,
                    batch_sampler = sampler,)  # DROP LAST?
            else:
                dataloaders[split] = Text2Shape_unique_shapes_in_batch_2(
                    device=device,  lang_model=t5,
                    lang_tokenizer=t5_tokenizer,  root=Path(dataroot),
                    chatgpt_prompts=True,  split=split,
                    categories=categories,  from_shapenet_v1=False,
                    from_shapenet_v2=False,  language_model="t5-11b",
                    lowercase_text=True,  max_length=max_length,
                    padding=False,  scale_mode="shapenet_v1_norm",
                    csv_alt_root=Path(tmp_csv_root),
                    batch_size = val_batch_size,
                    shorten=SHORTEN)
            
    humaneval_test = Text2Shape_humaneval(device=device,
        lang_model=t5,
        lang_tokenizer=t5_tokenizer,
        json_path=Path("./data/human_test_set_final.json"), # set proper cd in launch.json if using VScode
        t2s_root=Path(dataroot),
        categories=categories,
        from_shapenet_v1=False,
        from_shapenet_v2=False,
        language_model="t5-11b",
        lowercase_text=True,
        max_length=max_length,
        padding=False,
        scale_mode="shapenet_v1_norm")

    humaneval_dl = torch.utils.data.DataLoader(
        dataset=humaneval_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )
    
    del t5_tokenizer
    del t5
    torch.cuda.empty_cache()
    
    ## DEFINE MODELS
    reconstr_net = ReconstructionNet(device, model_type='reconstruction_pointnet2', \
        feat_dims=shape_embedding_dim, num_points=2048)
    ckpt = os.path.join(dataroot, 'autoencoder_weights', 'checkpoint.pth')
    ckpt = torch.load(ckpt, map_location=device)
    reconstr_net.load_state_dict(ckpt["model_state"])
    pointnet_color = reconstr_net.encoder 
    feats_dim = 1024
    mlp_decoder = MLPDecoder(feats_dim, [100, 50, 1], use_b_norm=True, dropout=False) # was: 0.4 or False

    # ------------------------------------------
    if BUILD_CACHE:
        #import gc
        SAVE_PATH = config.clouds_dir
        pointnet_color.to(device)
        found = {'train': set(), 'val': set()}
        for phase in ('train', 'val'):
            for samples in tqdm(dataloaders[phase], desc=f'building cache ({phase})'):
                pc_feats, _, _ = pointnet_color(samples['clouds'].to(device))  # [bsize, 128, 640]
                for i in range(len(pc_feats)):
                    shp = samples['original_id'][i]
                    if shp not in found[phase]:
                        fpath = os.path.join(SAVE_PATH, shp+'.pt')
                        found[phase].add(shp)
                        #if not os.path.exists(fpath):
                        torch.save(pc_feats[i], fpath)
        
        assert found['train'].isdisjoint(found['val'])
        
        for samples in tqdm(humaneval_dl, desc='building cache (human)'):
            pc_feats, _, _ = pointnet_color(samples['clouds'].view([batch_size*2, 2048, 6]).to(device))  # [bsize, 128, 640]
            for k in range(len(pc_feats)):
                mids = samples['mids']
                ids = [mids[j][i] for i in range(0, len(mids[0])) for j in (0,1) ]  # mids[0] and mids[1] interleaved
                fpath1 = os.path.join(SAVE_PATH, ids[k]+'.pt')
                if not os.path.exists(fpath1):
                    torch.save(pc_feats[k], fpath1)
        sys.exit(0)
    #'''
    # ------------------------------------------
    
    ########################################
    # If modified, change experiment name: #
    ########################################
    lr0 = 5.e-4      # original: 0.005
    lr_gamma = 0.95  # TR.2:0.95 #0.99
    EPOCHS_BEFORE_DECAY = 5
    USE_BT = True
    ########################################
    
    #listener = Attention_Listener_Spatial(pointnet_color, mlp_decoder, device, dim=640, n_heads=8, d_head=64, context_dim=text_embedding_dim, gated_ff=True, align=False, only_cross=True, no_residual=True, no_toout=True).to(device) # 640 when using color-aware PcEncoder, otherwise 512
    listener = Attention_Listener(
        config.clouds_dir,
        mlp_decoder,
        device,
        dim=640,
        n_heads=8,
        d_head=64,
        text_dim=text_embedding_dim,
        use_clip_loss = LOSS=='CLIP',
        t0b0 = (40., -0.2), # 1/batch_size ?
        ).to(device) # 640 when using color-aware PcEncoder, otherwise 512

    # define parameters to optimize: everything except for PcEncoder (text embeds have been computed offline by frozen T5)
    params = list(listener.logit_encoder.parameters()) +\
        list(listener.attn1.parameters()) +\
        list(listener.attn2.parameters()) +\
        list(listener.norm1.parameters()) +\
        list(listener.norm2.parameters()) +\
        ([listener.b, listener.t] if LOSS=='SIGMOID' and USE_BT else [])
    #params = list(listener.logit_encoder.parameters()) + list(listener.bit.parameters())
    print('parameters: ', len(params))
    
    optimizer = optim.Adam(params, lr=lr0)
    s1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=0)
    s2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[s1, s2], milestones=[EPOCHS_BEFORE_DECAY])
        
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=1e-5, mode='min', factor=.1)
    '''
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr0, #max lr
        epochs=train_epochs,
        steps_per_epoch=len(dataloaders['train']),
        pct_start=0.05,  # % of increasing time
        anneal_strategy='cos',
        #cycle_momentum=True,base_momentum=0.85,max_momentum=0.95,
        div_factor=10.,  # Determines the initial learning rate via init_lr=max_lr/div_factor Default:25
        final_div_factor=5.e2,  # Determines the min.lr via min_lr=init_lr/final_div_factor Default: 1e4
        # three_phase=False, last_epoch=-1, verbose=False
    )
    '''
    
    if LOSS=='CLIP':
        smoothed_cross_entropy = nn.CrossEntropyLoss(label_smoothing=.1)
    elif LOSS=='SIGMOID':
        log_sigmoid = nn.LogSigmoid()

    # we specify also torch.no_grad() when calling the PcEncoder inside Attention_Listener

    ## TRAIN THE MODEL
    #best_val_accuracy = 0.0
    #best_test_accuracy = 0.0
    #val_improved = False

    writer = SummaryWriter(output_dir)
    logger = setup_logging(output_dir)
    former_best_epoch_acc = -1.

    for epoch in range(1, train_epochs+1):
        for phase in splits:
            # TRAINING
            if phase == 'train':
                listener.train()
                bs = batch_size
            else:
                bs = val_batch_size
                listener.eval()

            labels = torch.tensor(range(bs), device=device)
            running_loss = 0.0
            running_corrects, running_corrects_txt, running_corrects_shp = 0, 0, 0
            n_batches = 0
            #running_corrects_human = 0
            
            # Avoid computing gradients of PcEncoder params
            #for param in listener.pc_encoder.net.parameters():
            #    param.requires_grad=False
            
            # check uniqueness:
            #for sample in dataloaders[phase]:
            #    if phase=='train' and (len(sample['original_id']) != len(set(sample['original_id']))):
            #        print('WARNING: non-unique batch while training.')
            
            #all_labels = torch.tensor(list(range(bs))*bs)
        
            if phase=='val':
                t_range = torch.tensor(range(bs)).to(device)
            # Iterate over data.
            for sample in tqdm(dataloaders[phase], desc=f"{phase}"):  # enumerate removed
                #start = datetime.now()
                n_batches += 1
                clouds = sample["clouds"] #[B, 2048, 6] - Unused from listener if using cache!
                text_embed = sample["text_embed"].to(device)    # [B, 77, 1024]                
                #text = sample["text"]
                
                with torch.set_grad_enabled(phase=='train'):
                    #clouds_shp = clouds.shape[1:]
                    #repeated_clouds = clouds.repeat(bs,1,1,1).permute(1,0,2,3).reshape(bs**2, *clouds_shp)
                    repeated_embeds = text_embed.repeat(bs,1,1)
                    
                    if phase=='train':
                        # To avoid the net learning identity (batch-aware permutation):                        
                        perm, new_col_labels, new_row_labels = permutation_indexes(bs, device)
                        # We don't need to in-batch permute "repeated_clouds", given that it's always the same cloud in-batch.
                        repeated_embeds = repeated_embeds[perm]
                        
                    ids = [_  for _ in sample['original_id'] for r in range(bs)]
                    #ids = [ids[i] for i in perm]  # permutation ADDED; should not matter with in-batch perm.
                        
                    logits = listener(
                        None,             # if uncached, use "repeated_clouds"
                        repeated_embeds,
                        ids = ids
                    ).view((bs, bs))  # row per each shape?
                    
                    if LOSS=='CLIP':
                        # .T is efficient, since it works on metadata
                        loss0 = smoothed_cross_entropy(logits,
                            new_row_labels if phase=='train' else t_range) # impl. reduce=None
                        loss1 = smoothed_cross_entropy(logits.T,
                            new_col_labels if phase=='train' else t_range)
                        loss =  (loss0+loss1)/2
                    elif LOSS=='SIGMOID':
                        # TODO: L2 NORMZ ?
                        # dot(zimg, ztxt.T)
                        if USE_BT:
                            logits = logits * torch.exp(listener.t) + listener.b
                        # -1 matrix, with diagonal 1, permuted:
                        labels = ((2 * torch.eye(bs, device=device) - torch.ones(bs, device=device)))
                        if phase=='train':
                            labels = labels[new_row_labels]
                        loss = -torch.sum(log_sigmoid(labels * logits)) / bs
                    else:
                        raise NotImplementedError
                        
                    #reg_loss = 0.0
                    #for p in listener.logit_encoder.parameters():
                    #    reg_loss += torch.sum(torch.abs(p))
                    #reg_loss *= reg_gamma                
                    #loss += reg_loss
                    
                    if phase=='train':
                        preds_ok_0 = torch.max(logits, dim=0)[1] == new_col_labels  # is each text associated to the right shape?
                        preds_ok_1 = torch.max(logits, dim=1)[1] == new_row_labels # is each shape associated to the right text
                    else:
                        preds_ok_0 = torch.max(logits, dim=0)[1] ==  t_range # is each text associated to the right shape
                        preds_ok_1 = torch.max(logits, dim=1)[1] == t_range # is each shape associated to the right text
                    
                    preds_ok = torch.logical_and(preds_ok_0, preds_ok_1).count_nonzero()
                    
                    if phase=='train':
                        # backward + optimize
                        listener.zero_grad()
                        loss.backward()
                        optimizer.step()
                        #writer.add_scalar(f'{phase}/lr', optimizer.param_groups[0]['lr'], epoch*len(dataloaders['train'])+(n_batches-1))
                        #get_lr_scheduler(epoch).step() # update lr at every STEP
                    '''
                    magnitudes = [round(torch.linalg.norm(_).item(), 3) for _ in (
                        listener.attn1.to_k.weight.grad,
                        listener.attn1.to_q.weight.grad,
                        listener.attn1.to_v.weight.grad,
                        listener.attn2.to_k.weight.grad,
                        listener.attn2.to_q.weight.grad,
                        listener.attn2.to_v.weight.grad,
                        listener.logit_encoder.net[-1].weight.grad,
                        #listener.logit_encoder.net[-1].bias.grad,
                    )]
                    for g in magnitudes:
                        if g < 1e-6:
                            print('WARNING: low grad', magnitudes, file=sys.stderr)
                            break
                    '''

                # statistics
                running_loss += loss.item() * bs  # --- CHECK ---
                running_corrects += torch.sum(preds_ok)
                running_corrects_txt += torch.sum(preds_ok_0)
                running_corrects_shp += torch.sum(preds_ok_1)
        
            '''
            tensor([[-1.1118, -1.0981, -1.1133, -1.1048],
        [-1.1014, -1.0976, -1.0884, -1.1121],
        [-1.1021, -1.1015, -1.1173, -1.0924],
        [-1.1094, -1.1152, -1.0903, -1.0911]], device='cuda:3',
       grad_fn=<DivBackward0>)
            '''
            n_examples = n_batches * bs  # ASSUMING drop_last=True
            logger.info(f'n_examples_{phase}: {n_examples}')
            epoch_loss = running_loss / n_examples
            epoch_acc = running_corrects.double() / n_examples
            epoch_txt_acc = running_corrects_txt.double() / n_examples
            epoch_shp_acc = running_corrects_shp.double() / n_examples
            logger.info(f'{epoch} Epoch  |  {phase} Loss:{epoch_loss:4.4f}  |  {phase} Acc:{epoch_acc:4.4f} (TXT_Acc:{epoch_txt_acc:4.4f}, SHP_Acc:{epoch_shp_acc:4.4f})')
            writer.add_scalar(f'{phase}/acc', epoch_acc, epoch)
            writer.add_scalar(f'{phase}/text_acc', epoch_txt_acc, epoch)
            writer.add_scalar(f'{phase}/shape_acc', epoch_shp_acc, epoch)
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            
            if phase=="train":
                lr_scheduler.step() # update lr at every epoch ; pass loss if needed
                writer.add_scalar(f'{phase}/lr', optimizer.param_groups[0]['lr'], epoch)
                if LOSS=='SIGMOID' and USE_BT:
                    writer.add_scalar(f'{phase}/sigmoid_b', listener.b, epoch)
                    writer.add_scalar(f'{phase}/sigmoid_t', listener.t, epoch)
                if 'val' not in splits:
                    # CHECKPOINT!
                    logger.info(f'Saving checkpoint in {output_dir} due to VAL. ACC. N/A')
                    save_dict = {
                        'epoch': epoch,
                        'model_state': listener.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    }
                    torch.save(save_dict, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
        
            if phase=='val':
                # CHECKPOINT!
                if epoch_acc > former_best_epoch_acc:
                    logger.info(f'Saving checkpoint in {output_dir} (best val. acc.)')
                    save_dict = {
                        'epoch': epoch,
                        'model_state': listener.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                        }
                    torch.save(save_dict, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
                    former_best_epoch_acc = epoch_acc
            
        # end for phase...

        # compute accuracy on Human Test Set
        logits, running_corrects_human = [], 0
        with torch.no_grad():
            print(end='Computing test on HST ds...', flush=True)
            tot = 0
            for i, sample in enumerate(humaneval_dl):
                listener.eval()
                clouds = sample["clouds"].to(device)
                target = sample["target"].to(device)    
                text_embed = sample["text_embed"].to(device)
                s1, sr = clouds.shape[0], clouds.shape[2:]   # assert clouds.shape[1]==2
                tot += s1
                mids = sample['mids']
                ids = [mids[j][i] for i in range(0, len(mids[0])) for j in (0,1) ]  # mids[0] and mids[1] interleaved
                logits = listener(
                    clouds.view(s1*2, *sr),
                    text_embed.repeat((2, *([1]*(len(text_embed.shape))))).permute(1,0,2,3).reshape(2*s1, *text_embed.shape[1:]),
                    ids=ids,
                ).view(-1,2)
                
                _, preds = torch.max(logits, 1)
                running_corrects_human += torch.sum(preds == target)
                
                if i<5 and (epoch-1)%5==0:  # print figure for first N in each batch mod5
                    visualize_pointcloud_pair(
                        *clouds[0].cpu(),
                        txt=sample['text'][0], 
                        filename=f'{epoch:3}_{i:2}.png',
                        cc1=logits[0][0], cc2=logits[0][1],
                        flip='xzy',
                    )
            
            #n_examples_human = len(humaneval_dl)*batch_size  # .dataset drop_last!
            epoch_acc_human = running_corrects_human.double() / tot
            print(end='\r')
            logger.info('%d Epoch |  Human Test Acc:%4.4f (%s examples) '% (epoch, epoch_acc_human, tot))
            writer.add_scalar(f'human_acc', epoch_acc_human, epoch)
        
        '''
        if phase == 'val':
            if not USE_CLIP_LOSS:
                n_examples_human = len(humaneval_dl)*batch_size  # .dataset drop_last!
                epoch_acc_human = running_corrects_human.double() / n_examples_human
                logger.info('n_examples_human: %s '%n_examples_human)
            
            if epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                val_improved = True
            else:
                val_improved = False
        
        # if we do not use test set, save checkpt when val improves
        if 'test' not in splits and phase=='val' and val_improved:
            logger.info('Saving checkpoint at epoch %d'  %epoch)
            save_dict = {
                'epoch': epoch,
                'model_state': listener.state_dict(),
                'optimizer_state': optimizer.state_dict()
                }
            torch.save(save_dict, os.path.join(output_dir, 'checkpoint_best.pth'))

        if 'test' in splits and phase == 'test' and val_improved:
            best_test_accuracy = epoch_acc
            # save model checkpoint
            logger.info('Saving checkpoint at epoch %d in %s'  %(epoch, output_dir))
            save_dict = {
                'epoch': epoch,
                'model_state': listener.state_dict(),
                'optimizer_state': optimizer.state_dict()
                }
            torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
                        
        if phase=='train':
            logger.info('%d Epoch  |  %s Loss:%4.4f  |  Acc:%4.4f'% (epoch, phase, epoch_loss, epoch_acc))
            #logger.info(f'LR is {lr_scheduler.get_last_lr()}')
        elif not USE_CLIP_LOSS:
            logger.info('%d Epoch  |  %s Loss:%4.4f  |  Acc:%4.4f  |  Human Test Acc:%4.4f '% (epoch, phase, epoch_loss, epoch_acc, epoch_acc_human))
            writer.add_scalar(f'test/human_acc', epoch_acc_human, epoch) # 6*
        else:
            logger.info('%d Epoch  |  %s Loss:%4.4f  |  Acc:%4.4f '% (epoch, phase, epoch_loss, epoch_acc))

        writer.add_scalar(f'{phase}/loss', epoch_loss, epoch) #6*
        writer.add_scalar(f'{phase}/acc', epoch_acc, epoch) #6*


    if 'test' in splits:
        logger.info('Best test so far: %4.4f'% best_test_accuracy)
    else:
        logger.info('Best val so far: %4.4f'% best_val_accuracy)
    '''

if __name__ == '__main__':
    main()

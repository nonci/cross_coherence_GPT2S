"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import sys
sys.path.append('../')
import torch
from torch import nn
import os
from datetime import datetime
from ..models.attention import CrossAttention, CrossAttention_Spatial, FeedForward
import numpy as np 
import random
import tqdm

class Listener(nn.Module):
    def __init__(self, language_encoder, image_encoder, mlp_decoder, pc_encoder=None):
        super(Listener, self).__init__()
        self.language_encoder = language_encoder
        self.image_encoder = image_encoder
        self.pc_encoder = pc_encoder
        self.logit_encoder = mlp_decoder

    def forward(self, item_ids, padded_tokens, dropout_rate=0.5):   # padded tokens: [B, 34]
        visual_feats = self.image_encoder(item_ids, dropout_rate)   # [B, <n_samples>, B]
        lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)  # lang_feats: list with 3 tensors of size [B, 100]

        if self.pc_encoder is not None:
            pc_feats = self.pc_encoder(item_ids, dropout_rate, pre_drop=False)      # pc_feats size [B, <n_samples>, 100]
        else:
            pc_feats = None

        logits = []
        for i, l_feats in enumerate(lang_feats):
            if pc_feats is not None:
                feats = torch.cat([l_feats, pc_feats[:, i]], 1) # feats size: [B, 200]
            else:
                feats = l_feats

            logits.append(self.logit_encoder(feats))    # logit encoder: MLP mapping from 200 to 100, then 50, then 1
        return torch.cat(logits, 1)                     # logits: list with 3 tensors of shape [B, 1]

class T2S_Listener(nn.Module):
    def __init__(self, pc_encoder, mlp_decoder, device):
        super(T2S_Listener, self).__init__()
        self.pc_encoder = pc_encoder
        self.logit_encoder = mlp_decoder
        self.device=device

    def forward(self, clouds, text_embed, text, dropout_rate=0.5):  # clouds:[B,2,2048,3], text_embed:[B,1024], text:[]
        
        sample_size = clouds.shape[1]   # n of samples (target + distractor)
    
         # save language features
        '''
        with open('metadata.tsv', "w") as f:
            f.write("{}\n".format(text))
        
        with open('text_vec.tsv', "w") as f:
            f.write("{}\t".format(text_embed))
        '''
        
        logits = []
        # get embedding of each point cloud
        for i in range(sample_size):
            start = datetime.now()
            pc_feats = self.pc_encoder(clouds[:,i], dropout_rate, pre_drop=False)   # input cloud: [B,2048,3]. output features: [B,1024]
            end = datetime.now()
            #print('time for computing shape features: ', (end-start).total_seconds(), ' s')
            # save pc features
            '''
            with open(f'pc_vec_{i}.tsv', "w") as f:
                f.write("{}\t".format(pc_feats))
            '''

            # CrossAttn
            # Pooling
            # Concat
            # new_feats

            feats = torch.cat([text_embed, pc_feats], 1)    # concatenate the same text embed with the features of the pcs
            
            start = datetime.now()
            logits.append(self.logit_encoder(feats))        # feats: [B, 2048] => logits: [B, 1], through 3 MLP layers (MLPDecoder)
            end = datetime.now()
            #print('time for applying MLP: ', (end-start).total_seconds(), ' s')
        return torch.cat(logits, 1)


'''
call_ctr = 0
def cuda_mem(device=3):
    global call_ctr
    call_ctr += 1
    #print('call', call_ctr, round(torch.cuda.max_memory_allocated(device)/2**30, 2))
    #print('call', call_ctr, round(torch.cuda.max_memory_reserved(device)/2**30,1))
    torch.cuda.reset_peak_memory_stats(device=device) 
    d = torch.cuda.memory_stats(device=device)
    #print('call', call_ctr, round(d["active_bytes.all.current"]/2**30, 2))
    print('call', call_ctr, round(d["inactive_split_bytes.all.current"]/2**30, 2))
    #torch.cuda.empty_cache()
    #print('call', call_ctr, round(torch.cuda.memory_reserved(device)/2**30,1))
'''

class Attention_Listener(nn.Module):
    def __init__(self, pc_encoder, mlp_decoder, device, dim, n_heads=None, d_head=None, dropout=0., text_dim=None,\
        use_clip_loss=False, t0b0 = (10., 0.), **stuff):
        super(Attention_Listener, self).__init__()
        self.pc_encoder = pc_encoder
        if type(self.pc_encoder) is str:
            # Let's load encoded clouds from disk
            self.cloud_emb_cache = dict()
            for f in tqdm.tqdm(os.listdir(self.pc_encoder), desc='loading clouds embedding cache'):
                self.cloud_emb_cache[f[:-3]] = torch.load(os.path.join(self.pc_encoder, f), map_location='cpu')
        
        self.logit_encoder = mlp_decoder
        #self.proj_feats = nn.Linear(context_dim, n_heads*d_head, bias=False)   # projection layer for global embeddings
        # context_dim = text embedding dim
        self.attn1 = CrossAttention(query_dim=dim, context_dim=text_dim,
                                    heads=n_heads, dim_head=d_head, inner_dim=n_heads*d_head, dropout=dropout)
        
        self.attn2 = CrossAttention(query_dim=text_dim, context_dim=dim,             # inputs are reversed
                                    heads=n_heads, dim_head=d_head, inner_dim=n_heads*d_head, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(text_dim)
        self.device=device
        self.use_clip_loss = use_clip_loss
        
        if not use_clip_loss:
            self.t = torch.nn.parameter.Parameter(data=torch.log10(torch.tensor(t0b0[0])), requires_grad=True)
            self.b = torch.nn.parameter.Parameter(data=torch.tensor(t0b0[1]), requires_grad=True)
        
        
    def forward2(self, clouds, text_embed, ids):
        #if clouds.shape[1] != 1: raise NotImplementedError # forward2 deals with clip_loss case
        
        if type(self.pc_encoder) is str:
            if ids:
                pc_feats = torch.stack([self.cloud_emb_cache[_] for _ in ids]).to(self.device) #[B**2,128,640]
            else:
                raise NotImplementedError
        else:
            raise Exception
            with torch.no_grad():
                self.pc_encoder.eval()
                #cuda_mem()
                points = clouds.to(self.device) #[:,0,:,:]
                #cuda_mem()
                pc_feats, _, _ = self.pc_encoder(points)   # when using pointnet_color # input cloud: [B,2048,6] output features: [B,128,640]

        #cuda_mem()
        if isinstance(pc_feats, tuple): raise NotImplementedError
        
        mask = text_embed!=0    # (B, seq_len, 1024)
        mask = mask[:,:,0]      # (B, seq_len): all the 1024 tensors have the same True and False values  

        x_1 = self.attn1(self.norm1(pc_feats), context=text_embed, mask=mask)       # x_1: Bx128x512
        x_2 = self.attn2(self.norm2(text_embed), context=pc_feats)                  # x_2: Bx(max_len)x512
    
        #cuda_mem()
        mean_x1 = torch.mean(x_1, dim=1)
        mean_x2 = torch.mean(x_2, dim=1)
        feats = torch.cat((mean_x1, mean_x2), dim=1)

        ef = self.logit_encoder(feats)        # feats: [B, 1024] => logits: [B, 1], through 3 MLP layers (MLPDecoder)
        #cuda_mem()
        return ef

    
    def forward(self, clouds, text_embed, ids):
        #if self.training:
        return self.forward2(clouds, text_embed, ids)
        #else:
        #return self.forward2(clouds, text_embed, class_labels, text, dropout_rate)
        
        

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, only_cross=True, no_residual=True, no_toout=True):
        super().__init__()

        self.only_cross = only_cross
        self.no_residual = no_residual
        self.no_toout = no_toout
        if not self.only_cross:
            self.attn1 = CrossAttention_Spatial(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, no_toout = self.no_toout) # is a self-attention
            self.norm1 = nn.LayerNorm(dim)
            if self.no_residual:
                if self.no_toout:
                    self.ff = FeedForward(n_heads*d_head, dropout=dropout, glu=gated_ff)
                    self.norm3 = nn.LayerNorm(n_heads*d_head)       
                    print('no to_out')   
                else:
                    self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
                    self.norm3 = nn.LayerNorm(dim)                    
            else:            
                self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
                self.norm3 = nn.LayerNorm(dim)
            if self.no_residual:
                if self.no_toout:
                    self.attn2 = CrossAttention_Spatial(query_dim=n_heads*d_head, context_dim=context_dim,
                                            heads=n_heads, dim_head=d_head, dropout=dropout, no_toout=self.no_toout)    
                    
                    self.norm2 = nn.LayerNorm(n_heads*d_head)   
                else:
                    self.attn2 = CrossAttention_Spatial(query_dim=dim, context_dim=context_dim,
                                            heads=n_heads, dim_head=d_head, dropout=dropout, no_toout=self.no_toout)

                    self.norm2 = nn.LayerNorm(dim)                
        else:
            self.attn2 = CrossAttention_Spatial(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, no_toout=self.no_toout)

            self.norm2 = nn.LayerNorm(dim)
            
    def forward(self, x, context=None, mask=None):
        if self.only_cross:
            x = self.attn2(self.norm2(x), context=context, mask=mask)
        elif self.no_residual:
            x = self.attn1(self.norm1(x))
            x = self.attn2(self.norm2(x), context=context, mask=mask)
            x = self.ff(self.norm3(x))
        else:
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
        
        return x

class Bilateral_Transformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, align=False, only_cross=True, no_residual=True, no_toout=True):
        super().__init__()
        self.block_pc = BasicTransformerBlock(dim, n_heads, d_head, dropout, context_dim, gated_ff, only_cross, no_residual, no_toout)
        self.block_text = BasicTransformerBlock(context_dim, n_heads, d_head, dropout, dim, gated_ff, only_cross, no_residual, no_toout)
        self.align = align
        if self.align:
            self.proj_input = nn.Linear(dim, 512)
            self.proj_context = nn.Linear(context_dim, 512)
    
    def forward(self, pc_feats, text_feats, mask):
        pc_feats_attn = self.block_pc(pc_feats, context=text_feats, mask=mask)           # [B,128,640]
        text_feats_attn = self.block_text(text_feats, context=pc_feats, mask=None)       # [B,77,1024]
        mean_pc = torch.mean(pc_feats_attn, dim=1)                                       # [B,640]
        mean_text = torch.mean(text_feats_attn, dim=1)                                   # [B,1024]
        if self.align:
            mean_pc = self.proj_input(mean_pc)                                      # [B,512]
            mean_text = self.proj_context(mean_text)                                # [B,512]
        feats = torch.cat((mean_pc, mean_text), dim=1)                              # [B,f1+f2]
        return feats


class Attention_Listener_Spatial(nn.Module):
    def __init__(self, pc_encoder, mlp_decoder, device, dim, n_heads=None, d_head=None, dropout=0., context_dim=None, gated_ff=True, align=False, only_cross=True, no_residual=True, no_toout=True):
        super(Attention_Listener_Spatial, self).__init__()
        self.pc_encoder = pc_encoder     
              
        self.logit_encoder = mlp_decoder

        self.bit = Bilateral_Transformer(dim, n_heads, d_head, dropout, context_dim, gated_ff, align, only_cross, no_residual, no_toout)
        
        self.device=device

    def forward(self, clouds, text_embed, class_labels, text, dropout_rate=0.5):
    
        sample_size = clouds.shape[1]   # n of samples (target + distractor)
        batch_size = clouds.shape[0]

        logits = []
        
        for i in range(sample_size):  
            self.pc_encoder.eval()                  
            points = clouds[:,i,:,:].to(self.device)     

            with torch.no_grad():
                seed = 2023
                # uncomment when testing AttentionGlot performance
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                pc_feats, pts, trans = self.pc_encoder(points)   # when using pointnet_color # input cloud: [B,2048,6] output features: [B,128,640]
                
                # apply inverse transformation
                x = torch.transpose(pts, 2, 1)          
                inv_trans = torch.inverse(trans)
                x = self.pc_encoder.tnet.transform(x, inv_trans, False, False)
                x = torch.transpose(x, 2, 1)
                pts_inv = x
        
            ### FOR DEBUGGING ###
            #torch.save(points, f'./visualization/ex_2/cloud_{i}.pt')
            #torch.save(pts, f'./visualization/ex_2/pts_128_{i}.pt')
            #torch.save(pts_inv, f'./visualization/ex_2/pts_128_inv_{i}.pt')

            # deal with case where pc_encoder gives local and global embeddings
            if isinstance(pc_feats, tuple):
                global_feats = pc_feats[0]      # (B, 1024)
                local_feats = pc_feats[1]       # (B, 128, 640)
                global_feats = self.proj_feats(global_feats)                # (B, 512)
                global_feats = global_feats.unsqueeze(1)                    # (B, 1, 512)                
                pc_feats = torch.cat((local_feats, global_feats), dim=1)    #  (B, 129, 640) localglobalcross
            

            mask = text_embed!=0    # (B, seq_len, 1024)
            mask = mask[:,:,0]      # (B, seq_len): all the 1024 tensors have the same True and False values  
            
            feats = self.bit(pc_feats, text_embed, mask)    # pc_feats=[B,128,640], text_embed=[B,77,1024], mask=[B,77]

            logits.append(self.logit_encoder(feats))        # feats: [B, 1024] => logits: [B, 1], through 3 MLP layers (MLPDecoder)
        return torch.cat(logits, 1)

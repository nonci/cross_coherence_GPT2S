"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from shapeglot.models.pointnet_partseg import Pointnet_encoder
from collections import OrderedDict

import copy
from pathlib import Path
from sched import scheduler
from typing import Any, Dict, Tuple

import numpy as np
import wandb
#from hesiod import get_out_dir, hcfg
from pycarus.datasets.shapenet_part import ShapeNetPartSegmentation
from pycarus.datasets.utils import get_shape_net_category_name_from_id
from pycarus.learning.models.dgcnn import DGCNNPartSegmentation
from pycarus.learning.models.pointnet import PointNetPartSegmentation
from pycarus.learning.models.pointnet2 import PointNet2PartSegmentationMSG
from pycarus.metrics.partseg_iou import PartSegmentationIoU
from pycarus.utils import progress_bar
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from pycarus.transforms.pcd import JitterPcd, RandomScalePcd

import sys, os
from inspect import getsourcefile
sys.path.append(os.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.sep)[:-2]))
from my_config import get_config
config = get_config()

class PretrainedFeatures(nn.Module):
    """
    Store a collection of features in a 2D matrix, that can be indexed and
    projected via an FC layer in a space with specific dimensionality.
    """
    def __init__(self, pretrained_features, embed_size):
        super(PretrainedFeatures, self).__init__()
        self.pretrained_feats = nn.Parameter(pretrained_features, requires_grad=False)
        self.fc = nn.Linear(self.pretrained_feats.shape[1], embed_size) # from 4096 to 100
        self.bn = nn.BatchNorm1d(embed_size)

    def __getitem__(self, index):
        assert index.ndim == 2
        res = torch.index_select(self.pretrained_feats, 0, index.flatten())
        res = res.view(index.size(0), index.size(1), res.size(1))
        return res

    def forward(self, index, dropout_prob=0.0, pre_drop=True):
        """ Apply dropout-fc-relu-dropout and return the specified by the index features.
        :param index: B x K
        :param dropout_prob:
        :param pre_drop: Boolean, if True it drops-out the pretrained feature before projection.
        :return: B x K x feat_dim
        """
        x = self[index]     # x size: ([B, 3, 4096])
        assert x.ndim == 3
        res = []
        for i in range(x.shape[1]):
            x_i = x[:, i]   # x_i size: [B, 4069]
            if pre_drop:
                x_i = F.dropout(x_i, dropout_prob, self.training)
            x_i = F.relu(self.fc(x_i), inplace=True)
            x_i = self.bn(x_i)
            x_i = F.dropout(x_i, dropout_prob, self.training)
            res.append(x_i)
        res = torch.stack(res, 1)
        return res

class PointCloudEncoder(nn.Module):
    """
    pre-trained Point Cloud Encoder, followed by projection layers
    """
    def __init__(self, pointnet_path, embed_size, device):
        super(PointCloudEncoder, self).__init__()
        
        # pre-trained PointNet encoder
        self.pretrained_pointnet = Pointnet_encoder(feat_dims=1024, device=device)
        ckpt = torch.load(pointnet_path)
        # filter weights before loading
        pretrained_weights = ckpt["state_dict"]
        new_pretrained_weights = OrderedDict()
        for k,v in pretrained_weights.items():
            k = k.replace("net.encoder.", "")   # remove the string "net.encoder."
            new_pretrained_weights[k] = v

        self.pretrained_pointnet.load_state_dict(new_pretrained_weights, strict=False)  # strict=False allows me to avoid loading decoder weights

        # projection layers
        #self.fc = nn.Linear(1024, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size)


    def forward(self, x, dropout_prob=0.0, pre_drop=True):
        """ Apply dropout-fc-relu-dropout and return the specified by the index features.
        :param index: B x K
        :param dropout_prob:
        :param pre_drop: Boolean, if True it drops-out the pretrained feature before projection.
        :return: B x K x feat_dim
        """
        assert x.ndim == 3                  # x:[B, 2048, 3]

        # get pc embedding
        x = self.pretrained_pointnet(x)     # x:[B, 1024, 1]     

        '''
        # Projection layer applied by ShapeGlot
        res = []
        for i in range(x.shape[1]):
            x_i = x[:, i]   # x_i size: [2048, 4069]
            if pre_drop:
                x_i = F.dropout(x_i, dropout_prob, self.training)
            x_i = F.relu(self.fc(x_i), inplace=True)
            x_i = self.bn(x_i)
            x_i = F.dropout(x_i, dropout_prob, self.training)
            res.append(x_i)
        res = torch.stack(res, 1)
        '''
        res = x.squeeze(-1)           # remove last dim = 1
        return res                      # res: [B,1024]

class LanguageEncoder(nn.Module):
    """
    Currently it reads the tokens via an LSTM initialized on a specific context feature and
    return the last output of the LSTM.
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    def __init__(self, n_hidden, embedding_dim, vocab_size, padding_idx=0):
        super(LanguageEncoder, self).__init__()
        # Whenever the embedding sees the padding index
        # it'll make the whole vector zeros
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(vocab_size,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, n_hidden, batch_first=True)

    def forward(self, padded_tokens, init_feats=None, drop_out_rate=0.5):
        w_emb = self.word_embedding(padded_tokens)
        w_emb = F.dropout(w_emb, drop_out_rate, self.training)
        len_of_sequence = (padded_tokens != self.padding_idx).sum(dim=1).cpu()
        x_packed = pack_padded_sequence(w_emb, len_of_sequence, enforce_sorted=False, batch_first=True)

        context_size = 1
        if init_feats is not None:
            context_size = init_feats.shape[1]  # context_size = 3 (n of samples = target + distractors)

        batch_size = len(padded_tokens)
        res = []
        for i in range(context_size):
            init_i = init_feats[:, i].contiguous()
            init_i = torch.unsqueeze(init_i, 0)    # rep-mat if multiple LSTM cells.
            rnn_out_i, _ = self.rnn(x_packed, (init_i, init_i))
            rnn_out_i, dummy = pad_packed_sequence(rnn_out_i, batch_first=True)
            lang_feat_i = rnn_out_i[torch.arange(batch_size), len_of_sequence - 1]
            res.append(lang_feat_i)
        return res

class PcdPartSegmenter:
    def __init__(self, ckpt_path, device) -> None:
        #dset_root = Path(hcfg("dset_root", str))
        #dset_root = Path("/media/data2/ldeluigi/datasets/ShapeNet/shapenet_partseg")
        dset_root = Path(config.shapenet_partseg)
        train_transforms = [
            RandomScalePcd(2.0 / 3.0, 3.0 / 2.0),
            JitterPcd(sigma=0.01, clip=0.05),
        ]
        
        train_dataset = ShapeNetPartSegmentation(root=dset_root, split="train", transforms=train_transforms)
        val_dataset = ShapeNetPartSegmentation(root=dset_root, split="val")
        test_dataset = ShapeNetPartSegmentation(root=dset_root, split="test")

        #train_bs = hcfg("train_bs", int)
        train_bs = 32
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_bs,
            shuffle=True,
            num_workers=8,
        )

        #val_bs = hcfg("val_bs", int)
        val_bs = 16
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=8,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=8,
        )
        
        self.device = device

        #self.num_points_pcd = hcfg("num_points_pcd", int)
        self.num_points_pcd = 2048
        #self.num_classes = hcfg("num_classes", int)
        self.num_classes = 16
        #self.num_part = hcfg("num_part", int)
        self.num_part = 50
        #net_name = hcfg("network", str)
        net_name = "pointnet2"

        if net_name == "dgcnn":
            net = DGCNNPartSegmentation(
                k=20,
                num_classes=self.num_classes,
                num_classes_part=self.num_part,
                use_tnet=False,
                use_geometry_nn=False,
                dropout_p=0.5,
                size_global_feature=1024,
            )

        if net_name == "pointnet2":
            net = PointNet2PartSegmentationMSG(
                use_tnet=False,
                num_classes=self.num_classes,
                num_classes_part=self.num_part,
                dropout_p=0.5,
                size_global_feature=1024,
            )

        if net_name == "pointnet":
            net = PointNetPartSegmentation(
                num_classes=self.num_classes,
                num_classes_part=self.num_part,
                use_pts_tnet=False,
                use_feats_tnet=False,
                size_global_feature=1024,
            )

        self.net = net.to(self.device)  # type: ignore

        #lr = hcfg("lr", float)
        lr = 0.001
        #wd = hcfg("wd", float)
        wd = 0.0001
        self.optimizer = AdamW(self.net.parameters(), lr, weight_decay=wd)
        #self.scheduler = CosineAnnealingLR(self.optimizer, hcfg("num_epochs", int), eta_min=1e-3)
        self.scheduler = CosineAnnealingLR(self.optimizer, 250, eta_min=1e-3)
        
        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

        #self.ckpts_path = get_out_dir() / "ckpts"
        self.ckpts_path = Path(ckpt_path).parent   # './partseg/ckpts/'

        #if self.ckpts_path.exists():
        #    self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)

    def get_one_hot_encoding(self, x: Tensor, num_classes: int) -> Tensor:
        one_hot = torch.eye(num_classes)[x.cpu()]
        one_hot = one_hot.to(x.device)

        return one_hot

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self) -> None:
        #num_epochs = hcfg("num_epochs", int)
        num_epochs = 250
        start_epoch = self.epoch
        self.val("val")

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            self.net.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                pcds, class_labels, part_labels = batch

                pcds = pcds.to(self.device)
                class_labels = class_labels.to(self.device)
                part_labels = part_labels.to(self.device)

                rand = torch.rand(pcds.shape[0], pcds.shape[1], device="cuda")
                batch_rand_perm = rand.argsort(dim=1)

                for idx in range(pcds.shape[0]):
                    pcds[idx] = pcds[idx, batch_rand_perm[idx], :]
                    part_labels[idx] = part_labels[idx, batch_rand_perm[idx]]

                pcds = pcds.contiguous()
                class_labels = self.get_one_hot_encoding(class_labels, self.num_classes)
                out = self.net(pcds, class_labels)[0]
                out = out.contiguous().view(-1, self.num_part)
                part_labels = part_labels.view(-1)
                loss = F.cross_entropy(out, part_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()

            if epoch == num_epochs - 1:
                self.val("test", best=True)

    @torch.no_grad()
    def val(self, split: str, best: bool = False) -> Tuple[Tensor, Tensor]:
        seg_classes = self.train_loader.dataset.class_to_parts
        seg_label_to_cat = {}

        for cat in seg_classes:
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        metric = PartSegmentationIoU(
            use_only_category_logits=True, category_to_parts_map=seg_classes
        )
        metric.reset()

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best:
            model = self.best_model
        else:
            model = self.net

        model = model.to("cuda")
        model.eval()
        losses = []
        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            pcds, class_labels, part_labels = batch     # pcds: [B, 2048, 3], class_labels: [16], part_labels: [16, 2048]
            pcds = pcds.to(self.device)
            class_labels = class_labels.to(self.device)
            part_labels = part_labels.to(self.device)
            pcds = pcds.contiguous()

            class_labels = self.get_one_hot_encoding(class_labels, self.num_classes)    # class_labels = [16, 16]
            seg_pred = self.net(pcds, class_labels)[0]
            out = seg_pred.contiguous().view(-1, self.num_part)
            losses.append(F.cross_entropy(out, part_labels.view(-1)))
            metric.update(seg_pred, part_labels)

        mIoU_per_cat, class_avg_iou, instance_avg_iou = metric.compute()
        self.logfn({f"{split}/class_avg_iou": class_avg_iou})
        self.logfn({f"{split}/instance_avg_iou": instance_avg_iou})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if split == "test":
            table = wandb.Table(columns=["class", "ioU"])
            for cat, miou in mIoU_per_cat.items():
                table.add_data(get_shape_net_category_name_from_id(cat), miou)
            wandb.log({"class IoU": table})

        if class_avg_iou > self.best_acc and split == "val":
            self.best_acc = class_avg_iou
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.net)

        return

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_acc": self.best_acc,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_acc = ckpt["best_acc"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
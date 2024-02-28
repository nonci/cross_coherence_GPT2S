import os
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from shapeglot.models.pointnet import PointNetEncoder
from shapeglot.models.pointnet2 import PointNet2EncoderMSG
from shapeglot.models.dgcnn import DGCNNEncoder


# data
from torch.utils.data import DataLoader, Subset
import sys
from data.text2shape_dataset import Text2Shape
from pathlib import Path

# utilities for chamfer function
from shapeglot.models.pointnet2_utils import batchify, unbatchify
from pytorch3d.ops.knn import knn_points  # type: ignore
from torch import Tensor, mean, sqrt
from typing import Tuple

# logging
import logging
from torch.utils.tensorboard import SummaryWriter
import wandb

from my_config import get_config
config_ = get_config()

def chamfer(
    predictions: Tensor,
    groundtruth: Tensor,
    squared: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the Chamfer Distance defined as in:

       Fan, H., Su, H., & Guibas, L. J. (2017).
       A point set generation network for 3d object reconstruction from a single image.
       In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 605-613).

    Args:
        prediction: The source point cloud(s) with shape ([B,] NUM_PTS_PRED, 3).
        groundtruth: The target point cloud(s) with shape ([B,] NUM_PTS_GT, 3).
        squared: If true return the squared euclidean distance.

    Returns:
        A tuple containing:
            - the chamfer distance with shape (B, ).
            - the accuracy, eg. point-wise distance from the prediction to the groundtruth with shape ([B,] NUM_PTS_PRED).
            - the completeness, eg. the distance from the groundtruth to the prediction with shape ([B,] NUM_PTS_GT).
            - the nearest neighbor indices from the prediction to the groundtruth with shape ([B,] NUM_PTS_PRED).
            - the nearest neighbor indices from the groundtruth to the prediction with shape ([B,] NUM_PTS_GT).
    """
    batched, [predictions, groundtruth] = batchify([predictions, groundtruth], 3)

    accuracy, indices_pred_gt, _ = knn_points(predictions, groundtruth)
    accuracy = accuracy.squeeze(-1)
    indices_pred_gt = indices_pred_gt.squeeze(-1)

    completeness, indices_gt_pred, _ = knn_points(groundtruth, predictions)
    completeness = completeness.squeeze(-1)
    indices_gt_pred = indices_gt_pred.squeeze(-1)

    if not squared:
        accuracy = sqrt(accuracy)
        completeness = sqrt(completeness)

    avg_accuracy = mean(accuracy, dim=-1)
    avg_completeness = mean(completeness, dim=-1)

    chamfer = avg_accuracy + avg_completeness

    if batched:
        chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred = unbatchify(
            [chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred]
        )

    return chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred


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

def visualize_evaluation(table, epoch, config, viz_loader, model, device):
    """Visualize validation result in a Weights & Biases Table"""
    
    gt_clouds, pred_clouds, chamfer_losses, w_rgb_losses = [], [], [], []
    
    for idx, sample in enumerate(tqdm(viz_loader, desc=f"Generating Visualizations for Epoch {epoch}/{config.train_epochs}")):
        gt_cloud = sample["pointcloud"].to(device)  # 1,2048,6

        model.eval()
        with torch.no_grad():
            features, pred_cloud = model(gt_cloud)

            # clip color values between 0 and 1
            clip_pred_cloud = pred_cloud.clone()
            clip_pred_cloud[:,:,3:] = torch.clamp(pred_cloud[:,:,3:], min=0.0, max=1.0)
            clip_pred_cloud = clip_pred_cloud.to(device)
            loss, chamfer_dist, weighted_rgb_loss, rgb_loss = get_loss(clip_pred_cloud, gt_cloud, rgb_weight=config.rgb_loss_weight)

        
        gt_cloud[:,:,3:] = (gt_cloud[:,:,3:]*255).to(int)
        gt_clouds.append(
            wandb.Object3D(torch.squeeze(gt_cloud, dim=0).cpu().numpy())
        )

        clip_pred_cloud[:,:,3:] = (clip_pred_cloud[:,:,3:]*255).to(int)
        pred_clouds.append(
            wandb.Object3D(torch.squeeze(clip_pred_cloud, dim=0).cpu().numpy())
        )

        chamfer_losses.append(chamfer_dist.item())

        w_rgb_losses.append(weighted_rgb_loss.item())

        for i in range(len(gt_clouds)):
            wandb.log({
                f"Clouds/gt_{i}": gt_clouds[i],
                f"Clouds/pred_{i}": pred_clouds[i],
                "epoch": epoch}
                )

    table.add_data(
        epoch, gt_clouds, pred_clouds, chamfer_losses, w_rgb_losses
    )
    return table

class Simple_Decoder(nn.Module):
    def __init__(self, feat_dims=1024, num_points=1024):
        super(Simple_Decoder, self).__init__()
        self.m = num_points
        self.folding1 = nn.Sequential(
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, self.m*6, 1),
        )
        
    def forward(self, x):
        print('decoder input: ', x.shape)
        x = self.folding1(x)                    # (batch_size, 6, num_points)
        x = x.reshape(-1, self.m, 6)            # (batch_size, num_points ,6)
        print('decoder output: ', x.shape)
        return x

class ReconstructionNet(nn.Module):
    def __init__(self, device, model_type, feat_dims=1024, num_points=1024):
        super(ReconstructionNet, self).__init__()
        self.model_type = model_type
        if model_type == "reconstruction_pointnet":
            self.encoder = PointNetEncoder(size_global_feature=feat_dims)
        elif model_type == "reconstruction_pointnet2":
            self.encoder = PointNet2EncoderMSG(channels_in=6, size_global_feature=feat_dims)
        elif model_type == "reconstruction_dgcnn":
            self.encoder = DGCNNEncoder(size_global_feature=feat_dims)
        else:
            raise Exception("No encoder found for {0}".format(model_type))
            
        self.decoder = Simple_Decoder(feat_dims, num_points)

    def forward(self, input):
        feature = self.encoder(input)[0]
        output = self.decoder(feature.unsqueeze(-1))
            
        return feature, output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

def get_loss(pred, gt, rgb_weight=0.025):
    '''
    compute loss between predicted and gt cloud
    '''

    # compute chamfer loss between xyz coords
    chamfer_dist, _, _, indices_pred_gt, indices_gt_pred = chamfer(pred[:,:,:3], gt[:,:,:3])
    chamfer_dist = chamfer_dist.mean()

    rgb_pred = pred[:,:,3:]
    indices_pred_gt = indices_pred_gt.unsqueeze(-1).repeat(1,1,3)
    rgb_gt = torch.gather(gt[:,:,3:], dim=1, index=indices_pred_gt)   # for each predicted point, I take the RGB of the closest in GT
    l1_loss = nn.L1Loss(reduction='mean')
    rgb_loss = l1_loss(rgb_pred, rgb_gt)

    loss = chamfer_dist + rgb_weight*rgb_loss

    return loss, chamfer_dist, rgb_weight*rgb_loss, rgb_loss

def main():

    # define logger params
    output_dir = './training_logs/clamping_embed2048/'
    logger = setup_logging(output_dir)
    wandb_project = "autoencoder-color"
    wandb_run_name = "clamping_embed2048"

    wandb.init(project=wandb_project, name= wandb_run_name, job_type='train')
    config = wandb.config
    config.num_workers = 1
    config.num_visualization_samples = 10
    config.val_frequency = 50

    # define table for shapes visualization
    table = wandb.Table(
                columns=[
                    "Epoch",
                    "GT Cloud",
                    "Pred Cloud",
                    "Chamfer Loss",
                    "Weighted RGB Loss"])

    # define embeddings dim
    config.embed_dim = 2048
    # define optimizer and loss params
    config.learning_rate = 1e-4
    config.train_epochs = 100000
    config.rgb_loss_weight = 0.025

    # define batch size
    config.batch_size = 40

    # instantiate dataset
    dataroot = Path(config_.t2s_dataset)
    dset    = {}
    dloader = {}
    for split in ['train', 'val', 'test']:
        dset[split] = Text2Shape(
                            root=dataroot,
                            chatgpt_prompts=False,
                            split = split,
                            categories = "all",
                            from_shapenet_v1 = False,
                            from_shapenet_v2 = False,
                            language_model = 't5-11b',
                            lowercase_text = True,
                            max_length = 77,
                            padding = False,
                            conditional_setup = False,
                            scale_mode = "shapenet_v1_norm")
        
        dloader[split] = DataLoader(dset[split], batch_size=config.batch_size, shuffle=split=="train")
    
        print(f'{split} Dataset Size: ', len(dset[split]))

    # load model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = ReconstructionNet(device, model_type='reconstruction_pointnet2', feat_dims=config.embed_dim, num_points=2048)
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)


    it=0
    train_loader = dloader['train']
    eval_loader = dloader['val']

    random_indices = random.sample(list(range(len(dset["val"]))), config.num_visualization_samples)

    vizualization_loader = DataLoader([dset["val"][idx] for idx in random_indices],
                                batch_size=1,
                                shuffle=False,
                                num_workers=config.num_workers)

    save_dict = {}
    best_eval_loss = 10.0
    for epoch in range(0, config.train_epochs):
        logger.info(f'Epoch {epoch}')

        # training step
        net.train()
        epoch_train_loss = 0.0
        for idx, sample in enumerate(tqdm(train_loader)):
            it +=1
            gt_cloud = sample["pointcloud"].to(device)
            with torch.autograd.set_detect_anomaly(True):
                features, pred_cloud = net(gt_cloud)
                pred_cloud=pred_cloud.to(device)

                # clip color values between 0 and 1
                clip_pred_cloud = pred_cloud.clone()
                clip_pred_cloud[:,:,3:] = torch.clamp(pred_cloud[:,:,3:], min=0.0, max=1.0)
                clip_pred_cloud = clip_pred_cloud.to(device)

                loss, chamfer_dist, weighted_rgb_loss, rgb_loss = get_loss(clip_pred_cloud, gt_cloud, rgb_weight=config.rgb_loss_weight)
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss.detach()
            epoch_train_loss += loss.item()

            wandb.log({
                "Train/Loss": loss,
                "Train/Chamfer": chamfer_dist,
                "Train/Weighted_RGB_loss": weighted_rgb_loss,
                "Train/RGB_loss": rgb_loss},
                step=it
            )

        num_train_examples = len(train_loader)
        epoch_train_loss = epoch_train_loss / num_train_examples
        
        wandb.log({
                "Train/EpochLoss": epoch_train_loss,
                "epoch": epoch
                })

        if epoch % config.val_frequency ==0:
            # evaluation step
            net.eval()
            with torch.no_grad():
                epoch_eval_loss = 0.0
                for idx, sample in enumerate(eval_loader):
                    gt_cloud = sample["pointcloud"].to(device)
                    features, pred_cloud = net(gt_cloud)
                    pred_cloud=pred_cloud.to(device)

                    # clip color values between 0 and 1
                    clip_pred_cloud = pred_cloud.clone()
                    clip_pred_cloud[:,:,3:] = torch.clamp(pred_cloud[:,:,3:], min=0.0, max=1.0)
                    clip_pred_cloud = clip_pred_cloud.to(device)

                    loss, chamfer_dist, weighted_rgb_loss, rgb_loss = get_loss(clip_pred_cloud, gt_cloud, rgb_weight=config.rgb_loss_weight)
                    loss = loss.mean()

                    loss.detach()
                    epoch_eval_loss += loss.item()

                    wandb.log({
                    "Eval/Loss": loss,
                    "Eval/Chamfer": chamfer_dist,
                    "Eval/Weighted_RGB_loss": weighted_rgb_loss,
                    "Eval/RGB_loss": rgb_loss},
                    step=it)

            num_eval_examples = len(eval_loader)
            epoch_eval_loss = epoch_eval_loss / num_eval_examples            

            wandb.log({
                "Eval/EpochLoss": epoch_eval_loss,
                "epoch": epoch
                })

            table = visualize_evaluation(table, epoch, config, vizualization_loader, net, device)

            wandb.log({"Eval Shapes": table})

            # Save the best model
            if epoch_eval_loss < best_eval_loss:
                best_eval_loss = epoch_eval_loss
                save_dict = {
                    'epoch': epoch,
                    'model_state': net.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                    }
                
                torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))

            logger.info(f'Epoch {epoch} | train loss: {epoch_train_loss} | eval loss: {epoch_eval_loss}')
        else:
            logger.info(f'Epoch {epoch} | train loss: {epoch_train_loss}')
        
        #lr_scheduler.step()


if __name__=="__main__":
    print('hello world')
    main()
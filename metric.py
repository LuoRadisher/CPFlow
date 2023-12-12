import torch
import numpy as np

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    EPS = 1e-6
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean


def cal_ade_mask(pred_traj,gt_traj,gt_vis,sample_flo_idx, non_sample_flo_idx):
    
    ate_metric = np.zeros((2,3))
    scene_abnormal = np.zeros(2)

    real_vis = torch.where(torch.sum(gt_vis[sample_flo_idx],0)==len(sample_flo_idx),1,0) #计算轨迹上完全可见的点
    real_occ = 1 - real_vis
    
    vis_point_idx = torch.where(real_vis==1)[0]
    occ_point_idx = torch.where(real_occ==1)[0]

    ate = torch.norm(pred_traj - gt_traj, dim=2)    # B,T,N
    gt_vis = gt_vis.unsqueeze(0)
    #real_vis = real_vis.unsqueeze(0).unsqueeze(1).repeat(1,ate.shape[1],1)
    #real_occ = real_occ.unsqueeze(0).unsqueeze(1).repeat(1,ate.shape[1],1)
    real_vis = gt_vis[:,:,vis_point_idx]
    real_occ = gt_vis[:,:,occ_point_idx]

    ate_vis = ate[:,:,vis_point_idx]
    ate_occ = ate[:,:,occ_point_idx]

    # ate_vis_sample = reduce_masked_mean(ate[:,sample_flo_idx],gt_vis[:,sample_flo_idx])
    # ate_vis_non_sample = reduce_masked_mean(ate[:,non_sample_flo_idx],gt_vis[:,non_sample_flo_idx])
    # ate_vis_all = reduce_masked_mean(ate,gt_vis)

    if len(vis_point_idx)!=0:

        ate_vis_sample = reduce_masked_mean(ate_vis[:,sample_flo_idx],real_vis[:,sample_flo_idx])
        ate_vis_non_sample = reduce_masked_mean(ate_vis[:,non_sample_flo_idx],real_vis[:,non_sample_flo_idx])
        ate_vis_all = reduce_masked_mean(ate_vis,real_vis)

        ate_metric[0,0] = ate_vis_sample.item()
        ate_metric[0,1] = ate_vis_non_sample.item()
        ate_metric[0,2] = ate_vis_all.item()
    
    if len(occ_point_idx) !=0:

        ate_occ_sample = reduce_masked_mean(ate_occ[:,sample_flo_idx],real_occ[:,sample_flo_idx])
        ate_occ_non_sample = reduce_masked_mean(ate_occ[:,non_sample_flo_idx],real_occ[:,non_sample_flo_idx])
        ate_occ_all = reduce_masked_mean(ate_occ,real_occ)

        ate_metric[1,0] = ate_occ_sample.item()
        ate_metric[1,1] = ate_occ_non_sample.item()
        ate_metric[1,2] = ate_occ_all.item()
    
    if len(vis_point_idx) == 0:
        scene_abnormal[0] = 1
    if len(occ_point_idx) == 0:
        scene_abnormal[1] = 1
    

    return ate_metric,scene_abnormal

def cal_ade(pred_traj,gt_traj,gt_vis,sample_flo_idx, non_sample_flo_idx):
    
    ate_metric = np.zeros(3)

    ate = torch.norm(pred_traj - gt_traj, dim=2)    # B,T,N
    gt_vis = gt_vis.unsqueeze(0)

    ate_vis_sample = reduce_masked_mean(ate[:,sample_flo_idx],gt_vis[:,sample_flo_idx])
    ate_vis_non_sample = reduce_masked_mean(ate[:,non_sample_flo_idx],gt_vis[:,non_sample_flo_idx])
    ate_vis_all = reduce_masked_mean(ate,gt_vis)

    ate_metric[0] = ate_vis_sample.item()
    ate_metric[1] = ate_vis_non_sample.item()
    ate_metric[2] = ate_vis_all.item()

    return ate_metric

def cal_RMSE(pred_flows , gt_flows ,gt_masks):
    
    rmse = 0

    point_num = gt_flows.shape[-1]
    gt_masks = gt_masks.unsqueeze(1).repeat(1,2,1)
    point_abnormal = 0 

    for point_idx in range(point_num):
        
        all_ssd = torch.sum(((pred_flows[:,:,point_idx] - gt_flows[:,:,point_idx])*gt_masks[:,:,point_idx])**2,1)#.sqrt()
        if torch.sum(gt_masks[:,0,point_idx]) == 0:
            point_abnormal += 1
            continue
        else:
            avg_ssd = torch.sum(all_ssd)/torch.sum(gt_masks[:,0,point_idx])
            rmse += (avg_ssd).sqrt()
    
    rmse = rmse/(point_num-point_abnormal)

    return rmse.cpu().item()
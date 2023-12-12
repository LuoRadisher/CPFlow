import tensorflow as tf
import pickle
import io
from PIL import Image
import numpy as np
from typing import Iterable, Mapping, Tuple, Union
import mediapy as media
import cv2
import torch
import numpy as np
import argparse
import os
import os.path as osp
from glob import glob
from raft_ode import CPflow 
from metric import cal_ade,cal_ade_mask,cal_RMSE
from core.build_curve import generate_bezier_param,resize_flow,b_spline
from core.utils.utils import InputPadder

from torch.utils.tensorboard import SummaryWriter

TRAIN_SIZE = (24, 256, 256, 3)

def fetch_flo_idx(seq_length,sample_frame=4):
    
    sample_time = sorted(list(np.linspace(0, seq_length-1,sample_frame,endpoint=True,dtype=np.int32)))
    all_time = sorted(list(np.linspace(0, seq_length-1 ,seq_length,endpoint=True,dtype=np.int32)))
    non_sample_time = sorted([t for t in all_time if t not in sample_time])
    
    sample_flo_idx = np.array(sample_time[1:]) - 1
    non_sample_idx = np.array(non_sample_time) - 1

    return (sample_flo_idx,non_sample_idx)


def fetch_timestamp(seq_length,time_type='all',sample_frame=4,add_sample=False):

    sample_time = sorted(list(np.linspace(0, seq_length-1,sample_frame,endpoint=True,dtype=np.int32)))
    all_time = sorted(list(np.linspace(0, seq_length-1 ,seq_length,endpoint=True,dtype=np.int32)))
    non_sample_time = sorted([t for t in all_time if t not in sample_time])

    sample_idx = np.array(sample_time)

    
    if time_type =='sample':
        norm_time = np.array([sample_time[1:]]) / (seq_length - 1)
        
    elif time_type == 'non_sample':
        norm_time =  np.array([non_sample_time]) / (seq_length - 1)
    
    elif time_type == 'all':
        norm_time = np.array([all_time[1:]]) / (seq_length - 1)
    
    if add_sample:
        sample_norm = torch.from_numpy(np.array(sample_time)/(seq_length -1)).unsqueeze(0)
        return (norm_time , sample_idx, sample_norm)
    else:
        return (norm_time , sample_idx)


def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    
    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64).cuda()*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B*H*W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N) # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output # B, C, N

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  #"""Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, TRAIN_SIZE[1:3])

def create_davis_dataset(davis_points_path: str):
  #"""Dataset for evaluating performance on DAVIS data."""
    pickle_path = davis_points_path

    with tf.io.gfile.GFile(pickle_path, 'rb') as f:
        davis_points_dataset = pickle.load(f)

    video_frames = []
    video_names =  []
    points_sets = []
    occ_sets = []

    for video_name in davis_points_dataset:
        frames = davis_points_dataset[video_name]['video']
        video_frames.append(resize_video(frames, TRAIN_SIZE[1:3]))   # T 256 256 3
        video_names.append(video_name) 
        target_points = davis_points_dataset[video_name]['points']  # N T 2
        target_occ = davis_points_dataset[video_name]['occluded']   # N T  occluded is 1
        target_points *= np.array([TRAIN_SIZE[2],TRAIN_SIZE[1],])
        points_sets.append(target_points)
        occ_sets.append(target_occ)

    davis_dataset = {'frames':video_frames,'points':points_sets,'occs':occ_sets,'video_name':video_names}
    return davis_dataset


def eval_kinetics_dataset(kinetics_path,args):

    output_list = []
    all_paths = tf.io.gfile.glob(osp.join(kinetics_path, '*_of_0010.pkl'))
    
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpus)
    device = torch.device('cuda')
    
    if args.model_encoder == 'ODE':
        model = torch.nn.DataParallel(CPflow(args))

    model.load_state_dict(torch.load(args.raw_model)['model_state_dict'])
    model = model.module
    model.to(device)
    model.eval()

    seq_length_range =  args.length
    eval_dict ={}

    H,W = args.raw_image_scale

    for seq_length in seq_length_range:

        sample_flo_idx,non_sample_flo_idx = fetch_flo_idx(seq_length)

        if args.curve_type == 'B-spline':
            cp_num = args.degree
            degree = args.pm
            nodevector = args.nodevector
            norm_time,frame_idx,sample_guide = fetch_timestamp(seq_length,args.time_type,args.sample_num,add_sample=True)
            eval_basis = b_spline(cp_num, degree , norm_time , nodevector).float().to(device)
        
        if args.method == 'ade_rmse':
            all_scene_loss = np.zeros(4)
            scene_abnormal = 0
        elif args.method == 'ade_occ':
            all_scene_loss = np.zeros((2,3))
            scene_abnormal = np.zeros(2)

        all_scene_length = 0
    
        for pickle_path in all_paths:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())

            print(pickle_path.split('/')[-1])
            # idx = random.randint(0, len(data) - 1)
            
            for idx in range(len(data)):

                example = data[idx]
                frames = example['video']
                if len(frames) < 250:
                    continue
                print(f'{idx} starting!')

                if isinstance(frames[0], bytes):
                    # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
                    def decode(frame):
                        byteio = io.BytesIO(frame)
                        img = Image.open(byteio)
                        return np.array(img)

                frames = np.array([decode(frame) for frame in frames[:seq_length]])

                if frames.shape[1] > TRAIN_SIZE[1] or frames.shape[2] > TRAIN_SIZE[2]:
                    frames = resize_video(frames, TRAIN_SIZE[1:3])

                frames = frames.astype(np.float32) #/ 255.0 * 2.0 - 1.0
                target_points = example['points'][:,:seq_length]
                target_occ = example['occluded'][:,:seq_length]
                target_points *= np.array([TRAIN_SIZE[2], TRAIN_SIZE[1]])
                
                with torch.no_grad():
                    clip_video = frames[:seq_length]
                    clip_traj = target_points[:,:seq_length]  # N T 2
                    clip_mask = target_occ[:,:seq_length]  # N T 
                    valid_point_index = np.where(clip_mask[:,0]==0)[0]  # get valid points at first frame
                    if len(valid_point_index) == 0:
                        continue
                    init_pos = target_points[valid_point_index,0]  
                    init_pos_tensor = torch.from_numpy(init_pos).unsqueeze(0).to(device) # B,N,2

                    seq_image = torch.from_numpy(clip_video[frame_idx]).permute(0,3,1,2).float()  #T C H W
                    correlation_block = seq_image
                    correlation_view = correlation_block.reshape(-1,seq_image.shape[-2],seq_image.shape[-1]).unsqueeze(0).to(device)
                
                    if args.method == 'ade_occ':
                        gt_traj = torch.from_numpy(clip_traj[valid_point_index,1:]).permute(1,2,0).to(device)

                    elif args.method == 'ade_rmse':
                        gt_traj = torch.from_numpy(clip_traj[valid_point_index,1:]).permute(1,2,0).to(device)
                        gt_flows = torch.from_numpy(clip_traj[valid_point_index,1:] - np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).to(device)

                    gt_visibles = torch.from_numpy(1-clip_mask[valid_point_index,1:]).permute(1,0).to(device)

                    if args.model_encoder == 'ODE':
                        _,flow_preds = model(correlation_view,eval_basis,sample_guide,norm_time,'B-spline',test_mode=True)   #B-spline测试
                        
                    flow_preds=flow_preds.reshape(1,-1,2,seq_image.shape[-2],seq_image.shape[-1])
                    mul_frame_flo=resize_flow(flow_preds[0],(H,W))

                    pred_flo = bilinear_sample2d(mul_frame_flo,init_pos_tensor[:,:,0],init_pos_tensor[:,:,1])  #T-1,2,N
                    
                    if args.method == 'ade_occ':
                        query_pos = torch.from_numpy(np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).float().to(device)
                        pred_traj = pred_flo + query_pos

                        pred_traj = pred_traj.unsqueeze(0)
                        gt_traj = gt_traj.unsqueeze(0)

                        metric,abnormal = cal_ade_mask(pred_traj,gt_traj,gt_visibles,sample_flo_idx,non_sample_flo_idx)
                        scene_abnormal += abnormal

                    elif args.method == 'ade_rmse':
                        query_pos = torch.from_numpy(np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).float().to(device)
                        pred_traj = pred_flo + query_pos

                        pred_traj = pred_traj.unsqueeze(0)
                        gt_traj = gt_traj.unsqueeze(0)

                        ade_metric = cal_ade(pred_traj,gt_traj,gt_visibles,sample_flo_idx,non_sample_flo_idx)
                        rmse_metric = cal_RMSE(pred_flo,gt_flows,gt_visibles)
                        metric = np.concatenate((ade_metric,np.array([rmse_metric])),0)

                    
                    print(f'{metric}')
                    scene_loss = metric
                    all_scene_loss += scene_loss
                    all_scene_length += 1
                    
                    # pickle_name = pickle_path.split('/')[-1]

                    # if args.save_flow and (seq_length == 48):
                    #     flo = pred_flo.cpu().numpy()
                    #     np.save(f'{args.output_dir}'+'/'+'kinetics'+'/'+'ode-6spline'+'/'+f'{pickle_name}_{idx}_{seq_length}.npy',flo)

        print(f'valid_scene_num:{all_scene_length}')
        #all_scene_metric = all_scene_loss/(all_scene_length-np.expand_dims(scene_abnormal,1).repeat(3,axis=1))
        all_scene_metric = all_scene_loss/(all_scene_length) 
        print(f'Total_loss of kinetics:{all_scene_metric}')
        

        corresponding_dict = ['sample','non_sample']

        if args.method == 'ade_occ':
            visible_dict = ['vis','occ']
            for i in range(2):
                for j in range(2):
                    eval_dict[f'{seq_length}_length_{corresponding_dict[i]}_ADE_{visible_dict[j]}'] = all_scene_metric[j][i]
        
        elif args.method == 'ade_rmse':

            for i in range(2):
                eval_dict[f'{seq_length}_length_{args.sample_num}_frame_{corresponding_dict[i]}_ade'] = all_scene_metric[i]
            eval_dict[f'{seq_length}_length_{args.sample_num}_frame_TRMSE'] = all_scene_metric[3]

    
    for key,value in eval_dict.items():
        print(f'{key}:{value}')

    return True


def evaluate_davis(args,target_dataset):
    
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpus)
    device = torch.device('cuda')
    
    if args.model_encoder == 'ODE':
        model = torch.nn.DataParallel(CPflow(args))

    model.load_state_dict(torch.load(args.raw_model)['model_state_dict'])
    model  = model.module
    model.to(device)
    model.eval()

    seq_length_range =  args.length
    eval_dict ={}

    H,W = args.raw_image_scale

    videos = target_dataset['frames']  # T H W 3
    gt_points = target_dataset['points'] # N T 2
    gt_masks = target_dataset['occs']    # N T
    video_names = target_dataset['video_name']
    
    dataset_length = len(videos)

    for seq_length in seq_length_range:
            
        sample_flo_idx,non_sample_flo_idx = fetch_flo_idx(seq_length,args.sample_num)
        

        # if args.curve_type == 'bezier':
        #     degree = args.degree
        #     norm_time,frame_idx = fetch_timestamp(seq_length,args.time_type,args.sample_num)
        #     eval_basis = generate_bezier_param(degree,norm_time).float().to(device)

        if args.curve_type == 'B-spline':
            cp_num = args.degree
            degree = args.pm
            nodevector = args.nodevector
            norm_time,frame_idx,sample_guide = fetch_timestamp(seq_length,args.time_type,args.sample_num,add_sample=True)
            eval_basis = b_spline(cp_num, degree , norm_time , nodevector).float().to(device)
        
        if args.method == 'ade_rmse':
            all_scene_loss = np.zeros(4)
            scene_abnormal = 0
        elif args.method == 'ade_occ':
            all_scene_loss = np.zeros((2,3))
            scene_abnormal = np.zeros(2)

    
        with torch.no_grad():
            for video_idx,video in enumerate(videos):

                print(f'No.{video_idx}:{video_names[video_idx]} starting!')
                print('-*'*25)
                print(f'{video_names[video_idx]}_length:{video.shape[0]}')
                print('-*'*25)
                
                if args.method == 'ade_rmse':
                    scene_loss = np.zeros(4)
                elif args.method == 'ade_occ':
                    scene_loss = np.zeros((2,3))

                clip_video = video[:seq_length]
                clip_traj = gt_points[video_idx][:,:seq_length]  # N T 2
                clip_mask = gt_masks[video_idx][:,:seq_length]  # N T 
                
                valid_point_index = np.where(clip_mask[:,0]==0)[0]  
                
                init_pos = gt_points[video_idx][valid_point_index,0]  
                init_pos_tensor = torch.from_numpy(init_pos).unsqueeze(0).to(device) # B,N,2

                seq_image = torch.from_numpy(clip_video[frame_idx]).permute(0,3,1,2).float()  #T C H W
                correlation_block = seq_image
                correlation_view = correlation_block.reshape(-1,seq_image.shape[-2],seq_image.shape[-1]).unsqueeze(0).to(device)
                #correlation_view = correlation_block.unsqueeze(0).to(device)
                
                if args.method == 'ade_occ':
                    gt_traj = torch.from_numpy(clip_traj[valid_point_index,1:]).permute(1,2,0).to(device)
                elif args.method == 'ade_rmse':
                    gt_traj = torch.from_numpy(clip_traj[valid_point_index,1:]).permute(1,2,0).to(device)
                    gt_flows = torch.from_numpy(clip_traj[valid_point_index,1:] - np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).to(device) #  T-1 2 N
                
                gt_visibles = torch.from_numpy(1-clip_mask[valid_point_index,1:]).permute(1,0).to(device)

                
                if args.model_encoder == 'ODE':
                    _,flow_preds = model(correlation_view,eval_basis,sample_guide,norm_time,'B-spline',test_mode=True)

                flow_preds=flow_preds.reshape(1,-1,2,seq_image.shape[-2],seq_image.shape[-1])

                mul_frame_flo=resize_flow(flow_preds[0],(H,W))
                pred_flo = bilinear_sample2d(mul_frame_flo,init_pos_tensor[:,:,0],init_pos_tensor[:,:,1])  #T-1,2,N
                
                
                if args.method == 'ade_rmse':
                    
                    query_pos = torch.from_numpy(np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).float().to(device)
                    pred_traj = pred_flo + query_pos

                    pred_traj = pred_traj.unsqueeze(0)
                    gt_traj = gt_traj.unsqueeze(0)

                    ade_metric = cal_ade(pred_traj,gt_traj,gt_visibles,sample_flo_idx,non_sample_flo_idx)
                    rmse_metric = cal_RMSE(pred_flo,gt_flows,gt_visibles)

                    metric = np.concatenate((ade_metric,np.array([rmse_metric])),0)

                    if ade_metric.all()==0:
                        scene_abnormal +=1
                
                elif args.method == 'ade_occ':
                    query_pos = torch.from_numpy(np.expand_dims(clip_traj[valid_point_index,0],1).repeat(seq_length-1,axis=1)).permute(1,2,0).float().to(device)
                    pred_traj = pred_flo + query_pos

                    pred_traj = pred_traj.unsqueeze(0)
                    gt_traj = gt_traj.unsqueeze(0)

                    metric,abnormal = cal_ade_mask(pred_traj,gt_traj,gt_visibles,sample_flo_idx,non_sample_flo_idx)
                    scene_abnormal += abnormal

                scene_loss = metric
                all_scene_loss += scene_loss
                print(f'{video_idx}_{args.method}:{scene_loss}')

                # if args.save_flow and (seq_length==24 or seq_length ==28 or seq_length == 32):
                #     flo = pred_flo.cpu().numpy()
                #     np.save(f'{args.output_dir}'+'/'+'davis'+'/'+'ode_6spline_img_uni_feat_uni_v2'+'/'+f'{video_idx}_{video_names[video_idx]}_{seq_length}_{args.sample_num}.npy',flo)

            if args.method == 'ade_occ':
                all_scene_metric = all_scene_loss/(dataset_length-np.expand_dims(scene_abnormal,1).repeat(3,axis=1))
            elif args.method == 'ade_rmse':
                all_scene_metric = all_scene_loss/(dataset_length-scene_abnormal)

            #avg_time = times.mean().item()
            
            print(f'Total_loss of DAVIS:{all_scene_metric}')
            
            
        corresponding_dict = ['sample','non_sample']

        if args.method == 'ade_occ':
            visible_dict = ['vis','occ']
            for i in range(2):
                for j in range(2):
                    eval_dict[f'{seq_length}_length_{corresponding_dict[i]}_ade_{visible_dict[j]}'] = all_scene_metric[j,i]
        
        elif args.method == 'ade_rmse':
            for i in range(2):
                eval_dict[f'{seq_length}_length_{args.sample_num}_frame_{corresponding_dict[i]}_ade'] = all_scene_metric[i]
            eval_dict[f'{seq_length}_length_{args.sample_num}_frame_TRMSE'] = all_scene_metric[3]

    
    for idx,(key,value) in enumerate(eval_dict.items()):
        print(f'{key}:{value}')

    return True
     

if __name__ == '__main__':
    
    parser=argparse.ArgumentParser()
    ##  Curve Setting
    parser.add_argument('--degree',default=6) 
    parser.add_argument('--pm',type=int,default=3)
    parser.add_argument('--nodevector',default=[0,0,0,0,1/3,2/3,1,1,1,1])
    parser.add_argument('--curve_type',default='B-spline')

    ##  Model Setting
    parser.add_argument('--raw_model',default='./checkpoints/cp_flow_30.pth')
    parser.add_argument('--model_encoder',default='ODE')
    parser.add_argument('--sample_num',default=4)
    parser.add_argument('--block_num',default=3)
    parser.add_argument('--init_dim',type=int,default=256)
    parser.add_argument('--n_layers',type=int,default=2)
    parser.add_argument('--input_size',default=(32,32))
    parser.add_argument('--run_backwards', action='store_true', default=True)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    ## Test setting
    parser.add_argument('--dataset_mode',default='kinetics') #'davis' 'kinetics'
    parser.add_argument('--raw_image_scale',default=(256,256))
    parser.add_argument('--method',default='ade_rmse')  # ade_occ or ade_rmse
    parser.add_argument('--time_type',default='all')
    parser.add_argument('--gpus',default=0)
    parser.add_argument('--local_rank',default=0)

    ## Output setting
    parser.add_argument('--output_dir',default='./save_flow')
    parser.add_argument('--save_flow',default=False)
    
    args = parser.parse_args()

    if args.dataset_mode == 'davis':
        dataset_root = './datasets/tap_vid_davis/tapvid_davis.pkl'
        davis_dataset = create_davis_dataset(dataset_root)
        args.length = [20, 24, 28, 32]
        evaluate_davis(args,davis_dataset)

    elif args.dataset_mode == 'kinetics':
        dataset_root = './datasets/tap_vid_kinetics/tap_kinetics_pickle'
        args.length = [36, 48, 128, 250]
        eval_kinetics_dataset(dataset_root,args)
    
    
    
    

        













    







    

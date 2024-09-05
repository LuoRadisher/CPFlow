import numpy as np
import os
#os.environ['MASTER_PORT'] = '8866'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,7'
from datetime import datetime
import os.path as osp
from glob import glob

#from bezier_model import RAFT as basic_RAFT
from raft_ode import CPflow as RAFT_ODE

from core.build_curve import generate_bezier_param,b_spline

from kubric_dataset import fetch_kubric_dataloader
from kubric_dataset import bilinear_sample2d


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank,destroy_process_group

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def set_ddp():
    init_process_group(backend='nccl', init_method='env://')    
    args.rank = get_rank()
    print(args.rank)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, int(args.total_num/(args.batch_size)*args.epoch),
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def sequence_loss(flow_preds, flow_gt, vis_gt ,query,curve_type,gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    #scene_idx = batch_idx
    B,T,C,N = flow_gt.shape
    device = flow_gt.device
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = bilinear_sample2d(flow_preds[i],query[:,0,0],query[:,0,1],device).reshape(B,-1,C,N)   #加遮挡
            
        temporal_loss = 0.0
        abnormal_ = 0
        for t in range(T):
            if curve_type == 'Bezier':
                t_loss = (flow_pred[:,t] - flow_gt[:,t]).abs()
            elif curve_type == 'B-spline':
                t_loss = (flow_pred[:,t+1] - flow_gt[:,t]).abs()
            
            if int(torch.sum(vis_gt[:,t])) != 0:
                basic_loss = torch.sum((t_loss * (vis_gt[:,t])))/(int(torch.sum(vis_gt[:,t]))+1e-8)

                if basic_loss != basic_loss:
                    abnormal_ += 1

                else:
                    temporal_loss += basic_loss
            else:
                abnormal_ += 1
                #torch.save(t_loss,f'/home/luojianqin/RAFT/RAFT-master/abnormal_case/t_loss_{scene_idx:05d}.pt')
        
        if curve_type =='B-spline':
            basic_start_loss = flow_pred[:,0].abs().mean()

            if basic_start_loss != basic_start_loss:
                abnormal_ += 1
            else:
                temporal_loss += basic_start_loss
            
        flow_loss += (i_weight * temporal_loss)/(T+1 - abnormal_+1e-8)
    
    flow_pred_final= bilinear_sample2d(flow_preds[-1],query[:,0,0],query[:,0,1],device).reshape(B,-1,C,N)
    #flo_index = np.linspace(1,flow_pred_final.shape[1],num=T,endpoint=False,dtype='int32')
    #flow = flow_pred_final[:,flo_index]
    Tepe = 0.0
    abnormal = 0
    for tao in range(T):
        if curve_type =='B-spline':
            if int(torch.sum(vis_gt[:,tao])) != 0:
                epe = (torch.sum(torch.sum(((flow_pred_final[:,tao+1]- flow_gt[:,tao])* (vis_gt[:,tao]))**2, dim=1).sqrt())/(int(torch.sum(vis_gt[:,tao]))+1e-8)).item()
            else:
                abnormal += 1
                continue    
                #torch.save(vis_gt[:,tao],f'/home/luojianqin/RAFT/RAFT-master/abnormal_case/tepe_{scene_idx:05d}.pt')
        else:
            if torch.sum(vis_gt[:,tao]) != 0:
                epe = (torch.sum(torch.sum(((flow_pred_final[:,tao]- flow_gt[:,tao])* (vis_gt[:,tao]))**2, dim=1).sqrt())/(int(torch.sum(vis_gt[:,tao]))+1e-8)).item()
            else:
                abnormal += 1
                continue       
        
        if epe != epe:
            abnormal += 1
        else:
            Tepe += epe
    
    metrics = Tepe/(T - abnormal + 1e-8)
    return flow_loss, metrics

def train(args):
    set_ddp()
    device = torch.device('cuda',args.local_rank)

    if args.encoder_type == 'ODE':
        if len(args.gpus)>1:
            model = nn.parallel.DistributedDataParallel(RAFT_ODE(args).to(device), device_ids=[args.local_rank],output_device=args.local_rank)
        else:
            model = RAFT_ODE(args)
    
    os.makedirs(os.path.join(args.save_path,args.subpath),exist_ok=True)
    
    model.train()


    train_scene_loader,args.total_num = fetch_kubric_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    
    if args.restore_ckpt is not None and 'full'  in args.restore_ckpt:
        print("Resume from checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, int(args.total_num/args.seq_batch*args.epoch),pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        initepoch = checkpoint['epoch'] + 1
        scheduler.last_epoch=638*checkpoint['epoch']
        print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
    elif args.restore_ckpt is not None and 'Bezier' in args.restore_ckpt:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('===>loaded pretrained model')
        initepoch = 1
    else:
        if args.rank == 0:
            print("====>no checkpoint found.")
        initepoch = 1
    
    scaler = GradScaler(enabled=args.mixed_precision)
    if args.rank == 0:
        writer=SummaryWriter('retrain_uniform_ode_spline6')

    #start = datetime.now()
    for epoch in range(initepoch,args.epoch+1):
        
        train_scene_loader.batch_sampler.set_epoch(epoch)   #batch_sampler

        model.train()
        if args.rank == 0:
            print(f'Epoch{epoch}/{args.epoch}')
            print('-'*10)

        epoch_loss = 0.0
        epoch_TEPE = 0.0
        #eval_epoch_tepe=0.0

        for batch_idx,batch_block in enumerate(train_scene_loader):

            if args.rank == 0:
                print(f'Batch_id:{batch_idx+1},start_training')
            
            optimizer.zero_grad()
            
            if args.encoder_type == 'ODE':                
                corr_view,flos,vis,query,norm_time,sample_time = batch_block
            elif args.encoder_type == 'Basic':
                corr_view,flos,vis,query,norm_time = batch_block
            
            B,_,H,W = corr_view.shape

            norm_index = norm_time.squeeze(1).numpy()

            if args.curve_type == 'Bezier':
                eval_basis=generate_bezier_param(args.degree,norm_index).float().to(device)
                
            elif args.curve_type == 'B-spline':
                eval_basis = b_spline(args.degree , args.pm , norm_index , args.nodevector).float().to(device)
            
            with torch.cuda.amp.autocast(enabled=args.mixed_precision): 

                if args.encoder_type == 'ODE':
                    flow_preds = model(corr_view.to(device),eval_basis,sample_time,norm_index,args.curve_type)
                elif args.encoder_type == 'Basic':
                    flow_preds = model(corr_view.to(device),eval_basis)
                
                loss , metrics= sequence_loss(flow_preds,flos.to(device),vis.to(device),query.to(device),args.curve_type,batch_idx)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            epoch_loss+=loss.item()
            epoch_TEPE+=metrics
        
        if args.rank == 0:              
            #print("Training complete in: " + str(datetime.now() - start))

            print(f'Average_Total_Loss:{epoch_loss/(batch_idx+1)}--epoch{epoch}')
            print(f'Average_Total_TEPE:{epoch_TEPE/(batch_idx+1)}--epoch{epoch}')
        
            writer.add_scalar(f'{args.name}_{args.degree}_epoch_Loss',epoch_loss/(batch_idx+1),epoch)
            writer.add_scalar(f'{args.name}_{args.degree}_TEPE',epoch_TEPE/(batch_idx+1),epoch)

            PATH = f'{args.save_path}/{args.subpath}/%02d_%s.pth' % (epoch, 'full')
            checkpoint = {"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),"epoch": epoch,"lr_schedule":scheduler.state_dict()}
            torch.save(checkpoint, PATH)
    
    destroy_process_group()

    #logger.close()
    PATH = f'{args.save_path}/{args.name}/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='ode_spline_kubric_sparse')
    parser.add_argument('--degree',default=6)
    parser.add_argument('--pm',type=int,default=3) # For B-spline
    parser.add_argument('--nodevector',default=[0,0,0,0,1/3,2/3,1,1,1,1])
    parser.add_argument('--curve_type',default='B-spline')
    parser.add_argument('--encoder_type',default='ODE')
    parser.add_argument('--mode',default='kubric')

    parser.add_argument('--crop_size',default=[128, 128])
    
    parser.add_argument('--dataset_dir',type=str,default=None)
    parser.add_argument('--subpath',default='ode_spline6')
    parser.add_argument('--sample_num',default = 4)
    parser.add_argument('--block_num',default = 3)
    parser.add_argument('--supervised_num',default= 8)
    parser.add_argument('--min_sample_length',default=8)
    parser.add_argument('--sparse_num',default=20480)
    #parser.add_argument('--stride',default=2)
    #parser.add_argument('--add_occlusion',default=True)

    parser.add_argument('--init_dim',type=int,default=256)
    parser.add_argument('--n_layers',type=int,default=2)
    parser.add_argument('--input_size',default=(32,32)) 
    parser.add_argument('--run_backwards', action='store_true', default=True)
    parser.add_argument('--dec_diff', type=str, default='dopri5', choices=['dopri5', 'euler', 'adams', 'rk4'])
    
    parser.add_argument('--small', default=False,action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', default=False,action='store_true', help='use mixed precision')
    parser.add_argument("--local_rank", type=int,help='rank in current node')                                           #

    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--batch_size',default=8)  
    parser.add_argument('--gpus',type=int, nargs='+',default=[0,1,2,3])
    parser.add_argument('--restore_ckpt',default=None)
    parser.add_argument('--lr', type=float, default=0.00002)
    
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')

    parser.add_argument('--save_path',type=str,default=None)
    
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])  # 1.12+高版本需要额外从环境变量导入local_rank
    torch.cuda.set_device(args.local_rank)
    train(args)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from core.update import BasicUpdateBlock, SmallUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
from core.base_conv_gru import *
from core.ode_func import ODEFunc, DiffeqSolver
from core.layers import create_convnet

try:
    autocast = torch.cuda.amp.autocast #半精度加速训练
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class CPflow(nn.Module):
    def __init__(self, args):
        super(CPflow, self).__init__()
        self.args = args
        self.block_num  = args.block_num
        self.degree= args.degree
        self.device = torch.device('cuda',args.local_rank)
        self.ode_dim = args.init_dim

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3 
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 3
            args.corr_radius = 3

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            
            self.block_fnet = BasicEncoder(1,output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(1,output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
    
        ##### NODE_1 + ConvGRU
        ode_func_netE = create_convnet(n_inputs=self.ode_dim,
                                        n_outputs=self.ode_dim,
                                        n_layers=self.args.n_layers,
                                        n_units=self.ode_dim // 2)
            
        rec_ode_func = ODEFunc(opt=self.args,
                                input_dim=self.ode_dim,
                                latent_dim=self.ode_dim,  # channels after encoder, & latent dimension
                                ode_func_net=ode_func_netE)
            
        z0_diffeq_solver = DiffeqSolver(self.ode_dim,
                                            ode_func=rec_ode_func,
                                            method="euler",
                                            latents=self.ode_dim,
                                            odeint_rtol=1e-3,
                                            odeint_atol=1e-4)
            
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=args.input_size,
                                                    input_dim=self.ode_dim,
                                                    hidden_dim=self.ode_dim,
                                                    kernel_size=(3, 3),
                                                    num_layers=1,
                                                    dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                    batch_first=True,
                                                    bias=True,
                                                    return_all_layers=True,
                                                    z0_diffeq_solver=z0_diffeq_solver,
                                                    run_backwards=self.args.run_backwards)
            
        ##### NODE_2
        ode_func_netD = create_convnet(n_inputs=self.ode_dim,
                                        n_outputs=self.ode_dim,
                                        n_layers=self.args.n_layers,
                                        n_units=self.ode_dim // 2)
            
        gen_ode_func = ODEFunc(opt=self.args,
                                input_dim=self.ode_dim,
                                latent_dim=self.ode_dim,
                                ode_func_net=ode_func_netD)
            
        self.diffeq_solver = DiffeqSolver(self.ode_dim,
                                            gen_ode_func,
                                            self.args.dec_diff, self.ode_dim,
                                            odeint_rtol=1e-3,
                                            odeint_atol=1e-4)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, T, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, T, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, T, 8*H, 8*W)

    def cal_cost_volume(self,fmap1,fmap2):
        "Calculate the Correlation Pyramids of frame pairs at different timestamps "

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, num_levels = self.args.corr_levels,radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels = self.args.corr_levels,radius=self.args.corr_radius)
        
        return corr_fn

    def forward(self, corr_block ,model_basis,truth_time_steps,time_steps_to_predict,curve_type,iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Generate continuous flow by estimating a set of control points for each pixel """
        
        corr_block = 2 * (corr_block / 255.0) - 1.0
        corr_block = corr_block.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        b,_,h,w = corr_block.shape
        pred_t_len = time_steps_to_predict.shape[-1]
        
        # In test mode, the uniform sampling is applied 
        if test_mode:
            cost_index = sorted(list(np.linspace(0, pred_t_len-1 , self.block_num, endpoint=True,dtype=np.int32)))

        # In training mode, the random sampling is applied
        else:
            if curve_type == 'Bezier':
                basic_index = sorted(list(np.linspace(0, pred_t_len-1 ,pred_t_len,endpoint=True,dtype=np.int32)))
            elif curve_type == 'B-spline':
                basic_index = sorted(list(np.linspace(0, pred_t_len-2 ,pred_t_len-1,endpoint=True,dtype=np.int32)))
            
            cost_index = sorted(random.sample(basic_index , self.block_num))  

        # ODE-ConvGRU encoder (NODE_1)
        time_steps_to_predict = torch.from_numpy(time_steps_to_predict)
        input_tensor = self.block_fnet(corr_block.reshape(-1,3,h,w)).view(b,-1,self.ode_dim,h//8,w//8) 
        first_point_mu, first_point_std = self.encoder_z0(input_tensor=input_tensor, time_steps=truth_time_steps)
        first_point_enc = first_point_mu.unsqueeze(0).repeat(1, 1, 1, 1, 1)                                      
        first_point_enc = first_point_enc.squeeze(0)
        
        # ODE decoder(NODE_2)
        sol_y = []
        for time_idx in range(b):
            sol_y.append(self.diffeq_solver(first_point_enc[time_idx].unsqueeze(0), time_steps_to_predict[time_idx]))
            
        #sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        sol_y = torch.cat(sol_y,0)
        sol_y = sol_y.contiguous().view(b, pred_t_len, -1, h // 8, w //8)

        # Build Multi-time Cost Volume
        corr_volume=[]
        img1 = corr_block[:,0:3]
        feat1 = input_tensor[:,0]
        with autocast(enabled=self.args.mixed_precision):
            for i in sorted(cost_index):  #range(pred_t_len)
                feat2 = sol_y[:,i]
                corr_volume.append(self.cal_cost_volume(feat1,feat2))

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(img1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(img1)
        #print(coords1.shape)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        coords=torch.cat([coords1]*self.degree,1)
        coords0=coords
        model_basis = model_basis.unsqueeze(1).repeat(1,coords0.shape[-2]*coords0.shape[-1],1,1)
        
        # Iterative Update Decoder 
        for itr in range(iters):
            coords = coords.detach()
            corr=[]
            for i in range(self.degree):
                for j in range(self.block_num):
                    corr.append(corr_volume[j](coords[:,i*2:i*2+2])) # index correlation volume
            cost=torch.cat(corr,1)
            control_L= coords - coords0
            
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_L = self.update_block(net, inp, cost, control_L)
                

            # F(t+1) = F(t) + \Delta(t)
            coords= coords+ delta_L
            temp_L = (coords - coords0).reshape(coords.shape[0],coords.shape[1],-1).permute(0,2,1)
            temp_L = temp_L.reshape(temp_L.shape[0] , temp_L.shape[1] , self.degree , 2)
            update_flow = torch.matmul(model_basis,temp_L).permute(0,2,3,1).reshape(coords.shape[0],model_basis.shape[2]*2,coords.shape[2],coords.shape[3])
            
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(update_flow)
            else:
                flow_up = self.upsample_flow(update_flow, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords - coords0, flow_up
            
        return flow_predictions

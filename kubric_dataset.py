import numpy as np
import os
import os.path as osp
import torch
import cv2
from torch.utils.data import Dataset,DataLoader
from glob import glob
from PIL import Image
from torchvision.transforms import ColorJitter
from torch.utils.data.distributed import DistributedSampler



class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(self, dataset ,batch_size):
        
        self.batch_size = batch_size
        self.data_source = dataset
        #self.num_replicas = 4
        
        rng = np.random.RandomState(seed=12)
        self.rng = rng
    
    def reset(self):
        """
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """

        # Record the lengths of each example.
        length_map = {}
        for i in range(len(self.data_source.scene_list)):
            x = self.data_source.scene_list[i]
            length = len(x[-1])
            length_map.setdefault(length, []).append(i)

        # Random Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            
            nbatches = len(arr) // (self.batch_size)#*self.num_replicas)
            surplus = nbatches * self.batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        '''if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]'''

        # Random select the length
        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        position = self.state[length]['position'] + 1
        start = position * self.batch_size
        batch_index = self.length_map[length][start: (start + self.batch_size)]#*self.num_replicas)]
        self.state[length]['position'] = position
        self.index = index
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)
    
def bilinear_sample2d(im, x, y, device,return_inbounds=False):
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

    base = torch.arange(0, B, dtype=torch.int64).to(device)*dim1
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


class KubricDataset(Dataset):

    def __init__(self, data_params ,dataset_location,crop_size = (128,128),save_ratio = 0.75, crop_save_ratio = 0.5):

        self.dataset_location = dataset_location
        self.sample_num = data_params['sample_num']             # sampled_num_frames
        #self.sample_range = data_params['sample_range']         
        self.supervised_num = data_params['supervised_num']     # sup_num for training
        self.min_sample_length = data_params['min_sample_length'] #mini_valid_length
        #assert self.sample_range_max == 8

        self.curve_type = data_params['curve_type']             # curve_type
        self.encoder_type = data_params['encoder_type']         # encoder_type
        self.sparse_num = data_params['sparse_num']
        

        self.crop_size = crop_size
        self.save_ratio = save_ratio
        self.crop_save_ratio = crop_save_ratio
        self.augmentor = Flow_augmentor_Kubric(self.crop_size, self.crop_save_ratio, self.sparse_num, do_flip=False,do_crop=False)
        self.all_vis = False 
        self.all_supervise = False
        
        self.scene_list = []

        color_ratio = [0.299,0.587,0.114]


        for scene_idx in sorted(os.listdir(self.dataset_location)[:200]):
            
            try:
                np.load(osp.join(self.dataset_location,scene_idx,'video.npy'))
            except:
                print(f'{scene_idx}导入异常')
                continue

            frames = np.array(np.load(osp.join(self.dataset_location,scene_idx,'video.npy')))         # num_frames,H,W,3 normlized to【-1，1】 
            occlusion = np.array(np.load(osp.join(self.dataset_location,scene_idx,'occlusion.npy')))  # num_points,num_frames

            T,H_raw,W_raw = frames.shape[:3]
            
            if np.max(np.sum((1 - occlusion),axis=0)) < self.sparse_num: 
                continue
                        
            visible_point_num = np.sum((1 - occlusion),axis=0)                                
            ref_candidate_idx = np.argmax(visible_point_num) 
            valid_length = (T - ref_candidate_idx)                                           
            if valid_length < self.min_sample_length:
                continue
            

            var_length = valid_length
            if self.all_supervise == False:
                supervised_num = self.supervised_num

            valid_frame_idx = np.linspace(ref_candidate_idx,(T-1),var_length,endpoint=True,dtype='int32')
            rest_frame_idx = valid_frame_idx[1:-1]   
            
            intensity_frames = np.sum(((frames[valid_frame_idx] + 1)/2.0)*255 * np.array(color_ratio),axis=3)
            intensity_diff = np.abs((np.log(1 + intensity_frames[1:-1]) - np.log(1 + intensity_frames[0])))

            intensity_avg = np.average(intensity_diff.reshape((var_length- 2),-1),axis=1)
            sample_prob = np.exp(intensity_avg)/(np.sum(np.exp(intensity_avg)))

            rest_sample_frame_idx = np.random.choice(rest_frame_idx, size = (self.sample_num - 2) ,replace=False , p=sample_prob)
            rest_sample_frame_idx = np.sort(np.append(rest_sample_frame_idx,(T-1)))
            sample_frame_idx = np.sort(np.append(rest_sample_frame_idx,ref_candidate_idx))
            
            if self.all_supervise == False:

                non_sample_candidate_idx = np.array([i for i in list(valid_frame_idx) if i not in list(sample_frame_idx)])                           
                non_sample_supervised_idx = np.random.choice(non_sample_candidate_idx,size= (supervised_num - self.sample_num),replace=False)   
                supervised_idx = np.sort(np.concatenate((rest_sample_frame_idx,non_sample_supervised_idx),0))                                 
            
            else:

                supervised_idx = valid_frame_idx[1:]
            
            self.scene_list.append((scene_idx,sample_frame_idx,supervised_idx))  

    def __getitem__(self, index):
        
        scene_idx, sample_frame_idx,supervised_idx= self.scene_list[index]

        seq_image_raw = np.array(np.load(osp.join(self.dataset_location,scene_idx,'video.npy')))
        seq_image = ((seq_image_raw + 1)/(2.0)) *255                          
        seq_traj = np.array(np.load(osp.join(self.dataset_location,scene_idx,'target.npy')))
        seq_occlusion = np.array(np.load(osp.join(self.dataset_location,scene_idx,'occlusion.npy')))

        seq_sample_image = seq_image[sample_frame_idx] 
        

        if self.all_vis:                       
            visible_point_idx = np.where(np.sum(seq_occlusion,axis=1)==0)[0]
        
        else:                                  
            visible_point_idx = np.where(seq_occlusion[:,sample_frame_idx[0]]==0)[0]

        query = np.expand_dims(seq_traj[visible_point_idx,sample_frame_idx[0]],1)   # N(vis) 1 2
        
        target = seq_traj[visible_point_idx]
        target = target[:,supervised_idx]
        
        seq_sample_occlusion = seq_occlusion[visible_point_idx]
        seq_sample_occlusion = seq_sample_occlusion[:,supervised_idx]
        
        seq_image_crop , query_crop , target_crop , seq_occ_crop =self.augmentor(seq_sample_image , query, target, seq_sample_occlusion)
        seq_flo_crop = target_crop - query_crop
        

        corr_torch=torch.from_numpy(seq_image_crop).permute(0,3,1,2).reshape(-1,seq_image_crop.shape[1],seq_image_crop.shape[2]).float()  # T*C H W 
        flo_torch =torch.from_numpy(seq_flo_crop).permute(1,2,0).float()                                # 7 2 N
        query_torch = torch.from_numpy(query_crop).permute(1,2,0).float()                              
        occ_torch = torch.from_numpy(seq_occ_crop).unsqueeze(1).permute(2,1,0).int()
        vis_torch = 1 - occ_torch                                                       # 7 1 N
        

        
        if self.curve_type == 'Bezier':
            norm_idx = (supervised_idx - sample_frame_idx[0]) / (23 - sample_frame_idx[0])
        elif self.curve_type == 'B-spline':
            norm_idx = (supervised_idx - sample_frame_idx[0]) / (23 - sample_frame_idx[0])
            norm_idx = np.insert(norm_idx,0,0)         
        
        if self.encoder_type == 'ODE':
            sample_idx = (sample_frame_idx - sample_frame_idx[0])  / (23 - sample_frame_idx[0])
            sample_int_index = (sample_frame_idx[1:] - sample_frame_idx[0])
            return [corr_torch, flo_torch , vis_torch , query_torch, norm_idx , sample_idx]
        elif self.encoder_type == 'Basic':
            return [corr_torch, flo_torch , vis_torch , query_torch, norm_idx ]
    
    def __len__(self):
        return len(self.scene_list)



class Flow_augmentor_Kubric:
    def __init__(self, crop_size, crop_save_ratio ,sparse_num ,do_flip=True,do_crop=False):
        
        # spatial augmentation params
        self.do_crop = do_crop
        self.crop_size = crop_size
        self.crop_save_ratio = crop_save_ratio

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.asymmetric_color_aug_prob = 0.5

        self.sparse_num = sparse_num

    
    def color_transform(self, imgs):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            imgs =np.array([np.array(self.photo_aug(Image.fromarray(np.uint8(imgs[i])))) for i in range(imgs.shape[0])])
        
        return imgs
    
    def spatial_transform(self,imgs,query,target,occs):

        '''if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                imgs = imgs[:,:, ::-1].copy()
                flos = flos[:,:, ::-1].copy() * [-1.0, 1.0]
                occs = occs[:,:, ::-1].copy()

            if np.random.rand() < self.v_flip_prob: # v-flip
                imgs = imgs[:,::-1, :].copy()
                flos = flos[:,::-1, :].copy() * [1.0, -1.0]
                occs = occs[:,::-1, :].copy()'''
        if self.do_crop:
            Excepted_Value = int(self.crop_save_ratio * (self.crop_size[0]*self.crop_size[1]))
            while True:
                y0 = np.random.randint(0, imgs.shape[1] - self.crop_size[0])
                x0 = np.random.randint(0, imgs.shape[2] - self.crop_size[1])
                y_coord = y0 + np.linspace(0,self.crop_size[0],self.crop_size[0],endpoint=False)
                x_coord = x0 + np.linspace(0,self.crop_size[1],self.crop_size[1],endpoint=False)
                in_range_points_num = np.sum(np.logical_and(np.in1d(np.around(query[:,0,0],0),x_coord),np.in1d(np.around(query[:,0,1],0),y_coord)))
                valid_point_idx = np.where(np.logical_and(np.in1d(np.around(query[:,0,0],0),x_coord),np.in1d(np.around(query[:,0,1],0),y_coord)))[0]
                if in_range_points_num >= (Excepted_Value):
                    final_point_idx = sorted(np.random.choice(valid_point_idx, size = Excepted_Value, replace=False))
                    break
            imgs = imgs[:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            query = query[final_point_idx]
            target = target[final_point_idx]
            occs = occs[final_point_idx]
        
        else:
            Excepted_Value = 20480 
            valid_point_idx = np.random.choice(range(0,target.shape[0],1),size = Excepted_Value, replace= False)
            query = query[valid_point_idx]
            target = target[valid_point_idx]
            occs = occs[valid_point_idx]
        
        return imgs,query,target,occs
    
    def __call__(self, imgs, query, target, occs):
        imgs = self.color_transform(imgs)
        imgs,query,target,occs = self.spatial_transform(imgs,query, target, occs)
        return imgs,query,target,occs

def fetch_kubric_dataloader(args):

    data_params={'sample_num':args.sample_num, 'supervised_num':args.supervised_num,'min_sample_length':args.min_sample_length,
    'curve_type':args.curve_type,'encoder_type':args.encoder_type,'sparse_num':args.sparse_num}
    dataset = KubricDataset(data_params,args.dataset_dir)
    total_samples = len(dataset)
    batch_sampler = DistributedBatchSamplerSimilarLength(dataset,args.batch_size)
    train_dataloader=DataLoader(dataset,batch_sampler=batch_sampler,num_workers=4,pin_memory=False)

    if args.rank==0:
        print(total_samples)

    return train_dataloader , total_samples

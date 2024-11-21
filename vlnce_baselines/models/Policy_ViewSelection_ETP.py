from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.etp.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.models.graph_utils import  MAX_DIST
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask, angle_feature_torch, pad_tensors, gen_seq_masks, get_angle_fts, get_angle_feature, get_point_angle_feature, calculate_vp_rel_pos_fts, calc_position_distance,pad_tensors_wgrad)
import math
from PIL import Image
import cv2
import open3d as o3d
from torch_kdtree import build_kd_tree
from vlnce_baselines.models.utils import  *

image_global_x_db = []
image_global_y_db = []
image_db=[]
DATASET = 'R2R'
# DATASET = 'RxR'
RGB_HW = 56

@baseline_registry.register_policy
class PolicyViewSelectionETP(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the ETP model ...')
        self.vln_bert = get_vlnbert_models(config=model_config)

        self.drop_env = nn.Dropout(p=0.4)

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        # if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
        #     self.rgb_encoder = TorchVisionResNet50(
        #         observation_space,
        #         model_config.RGB_ENCODER.output_size,
        #         device,
        #         spatial_output=model_config.spatial_output,
        #     )
        global DATASET
        if DATASET == 'R2R':
            self.rgb_encoder = CLIPEncoder(self.device,16)
        elif DATASET == 'RxR':
            self.rgb_encoder = CLIPEncoder(self.device,32)
            self.rxr_nerf_encoder = CLIPEncoder(self.device,16)

        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))


        batch_size = self.pano_angle_fts.shape[0]
        self.global_fts = [[] for i in range(batch_size)]
        self.global_position_x = [[] for i in range(batch_size)]
        self.global_position_y = [[] for i in range(batch_size)]
        self.global_position_z = [[] for i in range(batch_size)]
        self.global_patch_scales = [[] for i in range(batch_size)]
        self.global_patch_directions = [[] for i in range(batch_size)]
        self.global_mask = [[] for i in range(batch_size)]
        self.headings = [0 for i in range(batch_size)]
        self.positions = [0 for i in range(batch_size)]
        self.global_map_index = [[] for i in range(batch_size)]

        self.pointcloud_x = [[] for i in range(batch_size)]
        self.pointcloud_y = [[] for i in range(batch_size)]
        self.pointcloud_z = [[] for i in range(batch_size)]
        self.pointcloud_rgb = [[] for i in range(batch_size)]

        
        self.start_positions = None
        self.start_headings = None
        self.action_step = 0
        self.train()

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor
        global DATASET
        if DATASET == 'R2R':
            min_depth = 0.
            max_depth = 10.
        elif DATASET == 'RxR':
            min_depth = 0.5
            max_depth = 5.0

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        #mask2 = depth > 0.99
        #depth[mask2] = 0 # noise

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth

    def depth2image(self, depth):
        # depth - (B, H, W, 1) torch Tensor
        global DATASET
        if DATASET == 'R2R':
            min_depth = 0.
            max_depth = 10.
        elif DATASET == 'RxR':
            min_depth = 0.5
            max_depth = 5.0

        depth = (depth - min_depth) / (max_depth - min_depth)
        depth[depth>1.] = 1.
        depth[depth<0.] = 0.
        return depth

    def forward(self, mode=None, 
                txt_ids=None, txt_masks=None, txt_embeds=None, 
                batch_angle_idxes=None, batch_distance_idxes=None, observations=None, in_train=True,
                rgb_fts=None, dep_fts=None, loc_fts=None, 
                nav_types=None, view_lens=None,
                gmap_vp_ids=None, gmap_step_ids=None,
                gmap_img_fts=None, gmap_pos_fts=None, 
                gmap_masks=None, gmap_visited_masks=None, gmap_pair_dists=None, stepk=None, global_metric_num=None, delta_p=None, pred_pano_embeds = None): #num_st = None#new------------------------------------------------------------

        global DATASET, RGB_HW
        if mode == 'language':
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids, txt_masks,
            )
            return encoded_sentence
        

        elif mode == 'feature_field':
            # batch_size = observations['instruction'].size(0)
            batch_size = len(observations['rgb'])
            ''' encoding rgb/depth at all directions ----------------------------- '''
            if self.action_step == 1:
                NUM_IMGS = 12
            else:
                NUM_IMGS = 1

            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            if self.action_step == 1:
                # reverse the order of input images to clockwise
                a_count = 0
                for i, (k, v) in enumerate(observations.items()):
                    if 'depth'in k:  # You might need to double check the keys order
                        for bi in range(v.size(0)):
                            ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                            depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                            rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                        a_count += 1
            else:
                # reverse the order of input images to clockwise
                a_count = 0
                for i, (k, v) in enumerate(observations.items()):
                    if 'depth' == k:  # You might need to double check the keys order
                        for bi in range(v.size(0)):
                            ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                            depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                            rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                        a_count += 1

            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch
            #depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            #rgb_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7])

            #############################
            with torch.no_grad():
                if DATASET == 'R2R':
                    grid_fts = self.rgb_encoder(obs_view12, True)      # torch.Size([bs, 2048, 7, 7])
                elif DATASET == 'RxR':
                    grid_fts = self.rxr_nerf_encoder(obs_view12, True)      # torch.Size([bs, 2048, 7, 7])

            depth_batch_fts = torch.zeros((obs_view12['depth'].shape[0],14,14,1))
            for i in range(obs_view12['depth'].shape[0]):
                depth_batch_fts[i] = torch.tensor(cv2.resize(obs_view12['depth'][i].cpu().numpy(), (14, 14),  interpolation = cv2.INTER_NEAREST)).view(14,14,1)

            depth_batch_fts = self.preprocess_depth(depth_batch_fts).numpy()
            
            batch_grid_fts = grid_fts.view(batch_size,NUM_IMGS,197,512).cpu().numpy()
            
            for b in range(batch_size):
                position = {}
                position['x'] = self.positions[b][0]
                position['y'] = self.positions[b][-1]
                position['z'] = self.positions[b][1]
                heading = self.headings[b]
                depth = depth_batch[b]
                grid_ft = batch_grid_fts[b]
                self.getGlobalMap(b, position, heading, depth_batch_fts, grid_ft)

            #########################
            

        elif mode == 'waypoint':
            
            predicted_fts_view12 = []
            batch_size = len(observations['rgb'])
            with torch.no_grad():
                for b in range(batch_size):

                    selected_fts, (selected_position_x, selected_position_y, selected_position_z), selected_patch_directions, selected_patch_scales, fcd, fcd_tree, occupancy_pcd_tree = self.getSceneMemory(b)

                    for angle_id in range(12):
                        angle = (- math.pi / 6. * angle_id + self.headings[b]) % (2*math.pi)

                        position_dict = {}
                        position_dict['x'] = self.positions[b][0]
                        position_dict['y'] = self.positions[b][-1]
                        position_dict['z'] = self.positions[b][1]

                        scene_memory = (selected_fts, selected_patch_directions, selected_patch_scales, fcd, fcd_tree, occupancy_pcd_tree)
                            
                        predicted_fts = self.vln_bert.forward_nerf(scene_memory,(position_dict['x'],position_dict['y'],position_dict['z']), angle).detach() # freeze
                        predicted_fts_view12.append(predicted_fts[0:1])

            rgb_embedding = torch.cat(predicted_fts_view12,dim=0)

            ####################  Forward-facing CLIP features
            clip_fts = self.rgb_encoder(observations, False).detach()
            for b in range(batch_size):
                rgb_embedding[b*12] = clip_fts[b]

            ##########################

            # reverse the order of images back to counter-clockwise
            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, 12, 512, 1, 1)
            
            rgb_feats = torch.cat((
                rgb_embed_reshape[:,0:1,:], 
                torch.flip(rgb_embed_reshape[:,1:,:], [1]),
            ), dim=1)
           
            
            rgb_feats = self.space_pool_rgb(rgb_feats)
                     

            ''' waypoint prediction ----------------------------- '''

            # for cand
            cand_rgb = []
            
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):

                angle_idxes = batch_angle_idxes[j]
                distance_idxes = batch_distance_idxes[j]

                # for angle & distance
                angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # 顺时针
                angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
                cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append( (distance_idxes*0.25).tolist() )
                # for img idxes
                img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
                img_idxes[img_idxes==12] = 0
                cand_img_idxes.append(img_idxes)
                # for rgb & depth
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                
            
            # for pano
            pano_rgb = rgb_feats                            # B x 12 x 2048
         
            pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
            pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

            # cand_angle_fts 顺时针
            # cand_angles 逆时针
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]
             
                'cand_angle_fts': cand_angle_fts,   # [K x 4]
                'cand_img_idxes': cand_img_idxes,   # [K]
                'cand_angles': cand_angles,         # [K]
                'cand_distances': cand_distances,   # [K]

                'pano_rgb': pano_rgb,               # B x 12 x 2048
               
                'pano_angle_fts': pano_angle_fts,   # 12 x 4
                'pano_img_idxes': pano_img_idxes,   # 12 
            }
            return outputs

        elif mode == 'panorama':
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts, loc_fts, nav_types, view_lens, delta_p,
            )
            return outs

        elif mode == 'navigation':
            outs = self.vln_bert.forward_navigation(
                txt_embeds, txt_masks, 
                gmap_vp_ids, gmap_step_ids,
                gmap_img_fts, gmap_pos_fts, 
                gmap_masks, gmap_visited_masks, gmap_pair_dists, stepk,
                #num_st, #new------------------------------------------------------------
            )
            return outs
        
        elif mode == 'imagination':
            outs = self.vln_bert.forward_imagine(
                pred_pano_embeds,
            )
            return outs
        
        elif mode == 'prediction':
            outs = self.vln_bert.forward_predict(
                pred_pano_embeds,
            )
            return outs
        
        elif mode == 'observation':
            outs = self.vln_bert.forward_observe(
                pred_pano_embeds,
            )
            return outs

    def get_rel_position(self,depth_map,angle):
        global DATASET
        W=14
        H=14
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32) # / 4000.
        if DATASET == 'R2R':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi/4.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi/4.))
            scale = depth_y * math.tan(math.pi/4.) * 2. / W

        elif DATASET == 'RxR':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * 79./360.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * 79./360.))
            scale = depth_y * math.tan(math.pi * 79./360.) * 2. / W

        direction = (direction+angle) % (2*math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z
        return rel_x, rel_y, rel_z, direction.reshape(-1), scale.reshape(-1)

    def image_get_rel_position(self,depth_map,angle):
        global DATASET, RGB_HW
        W=RGB_HW
        H=RGB_HW
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32)
        if DATASET == 'R2R':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi/4)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi/4.))
        elif DATASET == 'RxR':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * 79./360.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * 79./360.))

        direction = (direction+angle) % (2*math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z

        return rel_x, rel_y, rel_z, direction.reshape(-1)

    def RGB_to_BGR(self,cvimg):
        pilimg = cvimg.copy()
        pilimg[:, :, 0] = cvimg[:, :, 2]
        pilimg[:, :, 2] = cvimg[:, :, 0]
        return pilimg



    def getSceneMemory(self, batch_id):

        i = batch_id
        #angle = - heading + math.pi
        
        # FeatureCloud
        global_position_x = np.concatenate(self.global_position_x[i],0)
        global_position_y = np.concatenate(self.global_position_y[i],0)
        global_position_z = np.concatenate(self.global_position_z[i],0)
        global_patch_scales = np.concatenate(self.global_patch_scales[i],0)
        global_patch_directions = np.concatenate(self.global_patch_directions[i],0)

        
        map_x = global_position_x
        map_y = global_position_y
        map_z = global_position_z

        fcd =  torch.tensor(np.concatenate((map_x.reshape((-1,1)),map_y.reshape((-1,1)),map_z.reshape((-1,1))),axis=-1),dtype=torch.float32).to("cuda")
        fcd_tree = build_kd_tree(fcd)


        global_position_x = global_position_x.reshape((-1,))
        global_position_y = global_position_y.reshape((-1,))
        global_position_z = global_position_z.reshape((-1,))
        
        global_fts = self.global_fts[i]

        #heading_angles = rectangular_to_polar(global_position_x,global_position_y)
        
        #select_ids = (global_position_y>0.) & (np.abs(np.degrees(heading_angles)) <= angular_range/2.) & (np.abs(global_patch_directions)<(math.pi/3.*2.))

        selected_fts = global_fts#[select_ids]
        selected_patch_scales = global_patch_scales#[select_ids]
        selected_patch_directions = global_patch_directions#[select_ids]

        selected_position_x = global_position_x#[select_ids]
        selected_position_y = global_position_y#[select_ids]
        selected_position_z = global_position_z#[select_ids]

        # Featurecloud Occupancy Map
        occupancy_pcd = torch.div(fcd,0.1, rounding_mode='floor')
        occupancy_pcd = torch.unique(occupancy_pcd, dim=0) * 0.1
        occupancy_pcd_tree = build_kd_tree(occupancy_pcd)

        return selected_fts, (selected_position_x, selected_position_y, selected_position_z), selected_patch_directions, selected_patch_scales, fcd, fcd_tree, occupancy_pcd_tree


    def getGlobalMap(self, batch_id, position, heading, depth_batch_fts, grid_ft):
        if self.action_step == 1:
            NUM_IMGS = 12
        else:
            NUM_IMGS = 1
        i = batch_id
        viewpoint_x_list = []
        viewpoint_y_list = []
        viewpoint_z_list = []
        viewpoint_scale_list = []
        viewpoint_direction_list = []
           

        depth = depth_batch_fts.reshape((depth_batch_fts.shape[0],-1))
        depth_mask = np.ones(depth.shape)
        depth_mask[depth==0] = 0
        self.global_mask[i].append(depth_mask)
       
        for ix in range(NUM_IMGS):
            rel_x, rel_y, rel_z, direction, scale = self.get_rel_position(depth[ix:ix+1],ix*math.pi/6-self.headings[i])  
            global_x = rel_x + position["x"]
            global_y = rel_y + position["y"]
            global_z = rel_z + position["z"]

            viewpoint_x_list.append(global_x)
            viewpoint_y_list.append(global_y)
            viewpoint_z_list.append(global_z)
            viewpoint_scale_list.append(scale)
            viewpoint_direction_list.append(direction)

        
        if self.global_fts[i] == []:
            self.global_fts[i] = grid_ft[:,1:].reshape((-1,512))
            
        else:
            self.global_fts[i] = np.concatenate((self.global_fts[i],grid_ft[:,1:].reshape((-1,512))),axis=0)

        position_x = np.concatenate(viewpoint_x_list,0)
        position_y = np.concatenate(viewpoint_y_list,0)
        position_z = np.concatenate(viewpoint_z_list,0)
        patch_scales = np.concatenate(viewpoint_scale_list,0)
        patch_directions = np.concatenate(viewpoint_direction_list,0)
        self.global_position_x[i].append(position_x)
        self.global_position_y[i].append(position_y)
        self.global_position_z[i].append(position_z)
        self.global_patch_scales[i].append(patch_scales)
        self.global_patch_directions[i].append(patch_directions)




class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


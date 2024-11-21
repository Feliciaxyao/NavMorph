
import numpy as np 
import torch
import torch.nn.functional as F


class SemanticGrid(object):
    
    def __init__(self, batch_size, grid_dim, crop_size, cell_size, spatial_labels, object_labels):
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.spatial_labels = spatial_labels
        self.object_labels = object_labels
        self.batch_size = batch_size
        self.crop_size = crop_size

        self.crop_start = int( (self.grid_dim[0] / 2) - (self.crop_size / 2) )
        self.crop_end = int( (self.grid_dim[0] / 2) + (self.crop_size / 2) )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # observed ground-projected sem grid over entire scene
        self.occupancy_grid = torch.ones((self.batch_size, self.spatial_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.occupancy_grid = self.occupancy_grid*(1/self.spatial_labels)

        self.semantic_grid = torch.ones((self.batch_size, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.semantic_grid = self.semantic_grid*(1/self.object_labels)


    def pop(self, batch_id):
        self.batch_size -= 1
        self.occupancy_grid = torch.cat([self.occupancy_grid[:batch_id],self.occupancy_grid[batch_id+1:]],dim=0)
        self.semantic_grid = torch.cat([self.semantic_grid[:batch_id],self.semantic_grid[batch_id+1:]],dim=0)


    # Transform each ground-projected grid into geocentric coordinates
    def spatialTransformer(self, grid, pose, abs_pose):
        # Input: 
        # grid -- sequence len x number of classes x grid_dim x grid_dim
        # pose -- sequence len x 3
        # abs_pose -- same as pose

        geo_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)


        for j in range(grid.shape[0]): # sequence length

            init_pose = abs_pose[j,0,:] # init absolute pose of each sequence
            init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                        [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)

            grid_step = grid[j,:,:,:].unsqueeze(0)
            pose_step = pose[j,:]
        
            rel_coord = torch.tensor([pose_step[1],pose_step[0]], dtype=torch.float32).to(grid.device)
            rel_coord = rel_coord.reshape((2,1))
            rel_coord = torch.matmul(init_rot_mat,rel_coord)
    
            x = 2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
            z = 2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
    
            angle = pose_step[2]

            trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
            rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
            trans_theta = trans_theta.to(grid.device)
            rot_theta = rot_theta.to(grid.device)
            
            trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
            rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
            
            rot_geo_grid = F.grid_sample(grid_step, rot_disp_grid.float(), align_corners=False ) # apply rotation
            geo_grid = F.grid_sample(rot_geo_grid, trans_disp_grid.float(), align_corners=False) # apply translation

            geo_grid = geo_grid + 1e-12
            geo_grid_out[j,:,:,:] = geo_grid

        return geo_grid_out

    
    # Transform a geocentric map back to egocentric view
    def rotate_map(self, grid, rel_pose, abs_pose):
            # grid -- sequence len x number of classes x grid_dim x grid_dim
            # rel_pose -- sequence len x 3
            ego_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)

            for i in range(grid.shape[0]): # sequence length

                init_pose = abs_pose[i,0,:] # init absolute pose of each sequence
                init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                                    [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)

                grid_step = grid[i,:,:,:].unsqueeze(0)
                rel_pose_step = rel_pose[i,:]
                rel_coord = torch.tensor([rel_pose_step[1],rel_pose_step[0]], dtype=torch.float32).to(grid.device)
                rel_coord = rel_coord.reshape((2,1))
                rel_coord = torch.matmul(init_rot_mat,rel_coord)
                x = -2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
                z = -2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
                angle = -rel_pose_step[2]
                
                trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
                rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
                trans_theta = trans_theta.to(grid.device)
                rot_theta = rot_theta.to(grid.device)
                
                trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
                rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
                trans_ego_grid = F.grid_sample(grid_step, trans_disp_grid.float(), align_corners=False ) # apply translation 
                ego_grid = F.grid_sample(trans_ego_grid, rot_disp_grid.float(), align_corners=False) # apply rotation
                ego_grid_out[i,:,:,:] = ego_grid
            return ego_grid_out
    

    def update_proj_grid_bayes(self, occup_grid=None, segm_grid=None):
        # Input geo_grid -- B x T (or 1) x num_of_classes x grid_dim x grid_dim
        # Update the ground-projected grid at each location4

        step_occup_grid = torch.zeros((occup_grid.shape[0], occup_grid.shape[1], self.spatial_labels, 
                                            self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(occup_grid.device)
        occup_grid = occup_grid.to(self.device)     
        for i in range(occup_grid.shape[1]): # sequence
            new_proj_grid = occup_grid[:,i,:,:,:]
            mul_proj_grid = new_proj_grid * self.occupancy_grid
            normalization_grid = torch.sum(mul_proj_grid, dim=1, keepdim=True)
            self.occupancy_grid = mul_proj_grid / normalization_grid.repeat(1, occup_grid.shape[2], 1, 1)
            step_occup_grid[:,i,:,:,:] = self.occupancy_grid.clone()


        step_segm_grid = torch.zeros((segm_grid.shape[0], segm_grid.shape[1], self.object_labels, 
                                            self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(segm_grid.device)
        segm_grid = segm_grid.to(self.device)     
        for i in range(segm_grid.shape[1]): # sequence
            new_proj_grid = segm_grid[:,i,:,:,:]
            mul_proj_grid = new_proj_grid * self.semantic_grid
            normalization_grid = torch.sum(mul_proj_grid, dim=1, keepdim=True)
            self.semantic_grid = mul_proj_grid / normalization_grid.repeat(1, segm_grid.shape[2], 1, 1)
            step_segm_grid[:,i,:,:,:] = self.semantic_grid.clone()

        return step_occup_grid, step_segm_grid


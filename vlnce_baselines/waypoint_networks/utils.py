import numpy as np
import os
import math
import torch
import quaternion
import torch.nn.functional as F
import gzip
import json
import random
import vlnce_baselines.waypoint_networks.viz_utils as viz_utils




def read_json_lines(filepath):
    data = []
    with gzip.open(filepath, "rt") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def locs_to_heatmaps(keypoints, img_size, out_size, sigma=1):
    x_scale = out_size[0]/img_size[0]
    y_scale = out_size[1]/img_size[1]
    x = torch.arange(0, out_size[1], dtype=torch.float32).to(keypoints.device)
    y = torch.arange(0, out_size[0], dtype=torch.float32).to(keypoints.device)
    yg, xg = torch.meshgrid(y,x)
    yg = yg.to(keypoints.device)
    xg = xg.to(keypoints.device)

    gaussian_hm = torch.zeros((keypoints.shape[0], out_size[0], out_size[1])).to(keypoints.device)
    for i,keypoint in enumerate(keypoints):
        kp_x = keypoint[0] * x_scale
        kp_y = keypoint[1] * y_scale
        gaussian_hm[i,:,:] = torch.exp(-((xg-kp_x)**2+(yg-kp_y)**2)/(2*sigma**2))
    return gaussian_hm


def heatmaps_to_locs(heatmaps, thresh=0):
    vals, uv = torch.max(heatmaps.view(heatmaps.shape[0], 
                                    heatmaps.shape[1], 
                                    heatmaps.shape[2]*heatmaps.shape[3]), 2)
    # zero out entries below the detection threshold
    uv *= (vals > thresh).type(torch.long)
    vals *= (vals > thresh).type(torch.long)
    rows = uv / heatmaps.shape[3]
    cols = uv % heatmaps.shape[3]
    return torch.stack([cols, rows], 2).cpu().type(torch.float), vals


def pck(gt_heatmaps, pred_heatmaps, visible=None):
    dist_thresh = gt_heatmaps.shape[2] / 5
    gt_locs, _ = heatmaps_to_locs(gt_heatmaps)
    pred_locs, _ = heatmaps_to_locs(pred_heatmaps)
    if visible is not None:
        visible = (visible > 0)
        return 100 * torch.mean((torch.sqrt(torch.sum((gt_locs - pred_locs) ** 2, dim=-1))[visible] < dist_thresh).type(torch.float))
    else:
        return 100 * torch.mean((torch.sqrt(torch.sum((gt_locs - pred_locs) ** 2, dim=-1)) < dist_thresh).type(torch.float))


def angle_diff(target_theta, pred_theta):
    return torch.abs(torch.atan2(torch.sin(target_theta-pred_theta), torch.cos(target_theta-pred_theta)))


def get_agent_location(pose):
    # Converting pose from RxR to our convention following utils.get_sim_location()
    # Here t[1] is not useful because the height does not correspond to the ground
    R = pose[:3,:3]
    t = pose[:3,3]
    x = -t[2]
    y = -t[0]
    height = t[1]
    quad = quaternion.from_rotation_matrix(R)
    axis = quaternion.as_euler_angles(quad)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(quad)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(quad)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height


def get_episode_pose(pose):
    # Convert pose from RxR to the episodes convention in R2R_VLNCE
    R = pose[:3,:3]
    t = pose[:3,3]
    quad = quaternion.from_rotation_matrix(R)
    return t, quad


def filter_pose_trace(poses_traced, return_idx=False):
    # There are a lot of almost duplicates in the pose trace
    # Keep next pose only if it's sufficiently different from previous
    poses_valid = []
    heights = []
    idx = []
    for i in range(poses_traced.shape[0]):
        pose_traced_0 = poses_traced[i]
        pose0, h = get_agent_location(pose_traced_0)
        pose0 = torch.tensor(pose0)
        if i==0:
            poses_valid.append(pose0)
            heights.append(h)
            idx.append(i)
        else:
            last_valid = poses_valid[-1]
            dist = torch.linalg.norm(pose0[:2] - last_valid[:2])
            angle_diff = wrap_angle(pose0[2]-last_valid[2])
            if dist > 0.1 or angle_diff > 0.1:
                poses_valid.append(pose0)
                heights.append(h)
                idx.append(i)
    if return_idx:
        return poses_valid, heights, idx
    return poses_valid, heights


def sample_waypoints(poses_valid, interval=0.1, num_waypoints=10):
    # Sample points between the pose coords
    # First interpolate between the points, then select K
    # Note: This function selects non-plausible poses that cannot be used in the simulation.
    #       Therefore this function is only used to generate the ground-truth for the first problem setup.
    #       To keep only valid poses, then omit this function.
    # interval: Distance between points during interpolation
    waypoints_all = torch.tensor([])
    for i in range(len(poses_valid)-1):
        p0 = poses_valid[i]
        p1 = poses_valid[i+1]
        n_y = torch.abs(p0[0]-p1[0])/interval
        n_x = torch.abs(p0[1]-p1[1])/interval
        if n_x >= n_y:
            num_p = int(n_x)
        else:
            num_p = int(n_y)
        interp_x = torch.linspace(p0[1], p1[1], num_p+1)
        interp_y = torch.linspace(p0[0], p1[0], num_p+1)
        interp_o = torch.linspace(p0[2], p1[2], num_p+1) # dummy angle interpolation
        if i < len(poses_valid)-2: # remove last point except in the last iteration
            interp_x = interp_x[:-1]
            interp_y = interp_y[:-1]
            interp_o = interp_o[:-1]
        for k in range(len(interp_o)):
            interp_o[k] = wrap_angle(interp_o[k])         
        
        points_tmp = torch.stack((interp_y, interp_x, interp_o), dim=1)

        if i==0:
            waypoints_all = points_tmp.clone()
        else:
            waypoints_all = torch.cat((waypoints_all, points_tmp.clone()), dim=0)

    if waypoints_all.shape[0]<2:
        return None

    # Sample k waypoints to use as ground-truth
    k = math.ceil(waypoints_all.shape[0] / (num_waypoints-1))
    waypoints = waypoints_all[::k] # need to verify this always gives num_waypoints
    waypoints = torch.cat((waypoints, waypoints_all[-1].view(1,-1)), dim=0) # add last element
    
    while waypoints.shape[0] < num_waypoints:
        rand_idx = random.randint(1, len(waypoints_all)-1)
        waypoints = torch.cat((waypoints, waypoints_all[-1].view(1,-1)), dim=0) # add last element again
    while waypoints.shape[0] > num_waypoints:
        mid = int(waypoints.shape[0]/2) # remove mid element
        waypoints = torch.cat((waypoints[:mid], waypoints[mid+1:]), dim=0)
    return waypoints


def wrap_angle(o):
    # convert angle to -pi,pi range
    if o < -math.pi:
        o += 2*math.pi
    if o > math.pi:
        o -= 2*math.pi
    return o


def add_uniform_noise(tensor, a, b):
    return tensor + torch.FloatTensor(tensor.shape).uniform_(a, b).to(tensor.device)

def add_gaussian_noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()).to(tensor.device) * std + mean

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)


def preprocess_img(img, cropSize, pixFormat, normalize):
    img = img.permute(0,3,1,2).float()
    img = F.interpolate(img, size=cropSize, mode='bilinear', align_corners=True)
    if normalize:
        img = img / 255.0
    return img


# normalize code from habitat lab:
# obs = (obs - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
def unnormalize_depth(depth, min, max):
    return (depth * (max - min)) + min


def get_entropy(pred):
    log_predictions = torch.log(pred)
    mul_map = -pred*log_predictions
    return torch.sum(mul_map, dim=2, keepdim=True) # B x T x 1 x cH x cW



def get_sim_location(agent_state):
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    height = agent_state.position[1]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height


def get_rel_pose(pos2, pos1):
    x1, y1, o1 = pos1
    if len(pos2)==2: # if pos2 has no rotation
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        x2, y2, o2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        do = o2 - o1
        if do < -math.pi:
            do += 2 * math.pi
        if do > math.pi:
            do -= 2 * math.pi
        return dx, dy, do


def load_scene_pcloud(preprocessed_scenes_dir, scene_id, n_object_classes):
    pcloud_path = preprocessed_scenes_dir+scene_id+'_pcloud.npz'
    if not os.path.exists(pcloud_path):
        raise Exception('Preprocessed point cloud for scene', scene_id,'not found!')

    data = np.load(pcloud_path)
    x = data['x']
    y = data['y']
    z = data['z']
    label_seq = data['label_seq']
    data.close()

    label_seq[ label_seq<0.0 ] = 0.0
    # Convert the labels to the reduced set of categories
    label_seq_spatial = label_seq.copy()
    label_seq_objects = label_seq.copy()
    for i in range(label_seq.shape[0]):
        curr_lbl = label_seq[i,0]
        label_seq_spatial[i] = viz_utils.label_conversion_40_3[curr_lbl]
        label_seq_objects[i] = viz_utils.label_conversion_40_27[curr_lbl]
    return (x, y, z), label_seq_spatial, label_seq_objects


def load_scene_color(preprocessed_scenes_dir, scene_id):
    # loads the rgb information of the map
    color_path = preprocessed_scenes_dir+scene_id+'_color.npz'
    if not os.path.exists(color_path):
        raise Exception('Preprocessed color for scene', scene_id,'not found!')

    data = np.load(color_path)
    r = data['r']
    g = data['g']
    b = data['b']
    color_pcloud = np.stack((r,g,b)) # 3 x Npoints
    return color_pcloud


def depth_to_3D(depth_obs, img_size, xs, ys, inv_K):

    depth = depth_obs[...,0].reshape(1, img_size[0], img_size[1])

    # Unproject
    # negate depth as the camera looks along -Z
    # SPEEDUP - create ones in constructor
    xys = torch.vstack((torch.mul(xs, depth) , torch.mul(ys, depth), -depth, torch.ones(depth.shape, device='cuda'))) # 4 x 128 x 128
    xys = xys.reshape(4, -1)
    xy_c0 = torch.matmul(inv_K, xys)

    # SPEEDUP - don't allocate new memory, manipulate existing shapes
    local3D = torch.zeros((xy_c0.shape[1],3), dtype=torch.float32, device='cuda')
    local3D[:,0] = xy_c0[0,:]
    local3D[:,1] = xy_c0[1,:]
    local3D[:,2] = xy_c0[2,:]

    return local3D



def run_img_segm(model, input_batch, object_labels, crop_size, cell_size, xs, ys, inv_K, points2D_step):
    
    pred_img_segm = model(input_batch)
    
    # get labels from prediction
    img_labels = torch.argmax(pred_img_segm['pred_segm'].detach(), dim=2, keepdim=True) # B x T x 1 x cH x cW
    
    # ground-project the predicted segm
    depth_imgs = input_batch['depth_imgs']
    
    pred_ego_crops_sseg = torch.zeros((depth_imgs.shape[0], depth_imgs.shape[1], object_labels,
                                                    crop_size[0], crop_size[1]), dtype=torch.float32, device=depth_imgs.device)
    
    for b in range(depth_imgs.shape[0]): # batch size
        
        points2D = []
        local3D = []
        for i in range(depth_imgs.shape[1]): # sequence

            depth = depth_imgs[b,i,:,:,:].permute(1,2,0)
            local3D_step = depth_to_3D(depth, img_size=(depth.shape[0],depth.shape[1]), xs=xs, ys=ys, inv_K=inv_K)

            points2D.append(points2D_step)
            local3D.append(local3D_step)
        
        pred_ssegs = img_labels[b,:,:,:,:]

        # use crop_size directly for projection
        pred_ego_crops_sseg_seq = ground_projection(points2D, local3D, pred_ssegs,
                                                            sseg_labels=object_labels, grid_dim=crop_size, cell_size=cell_size)
        
        pred_ego_crops_sseg[b,:,:,:,:] = pred_ego_crops_sseg_seq

       
    return pred_ego_crops_sseg, pred_img_segm['pred_segm'].squeeze(0)


# Taken from: https://github.com/pytorch/pytorch/issues/35674
def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode='floor')

    return coord.flip(-1)



import numpy as np
import os
import torch
from .semantic_grid import SemanticGrid




def est_occ_from_depth(local3D, grid_dim, cell_size, device, occupancy_height_thresh=-0.9):

    ego_grid_occ = torch.zeros((len(local3D), 3, grid_dim[0], grid_dim[1]), dtype=torch.float32, device=device)

    for k in range(len(local3D)):

        local3D_step = local3D[k]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]

        # initialize all locations as unknown (void)
        occ_lbl = torch.zeros((local3D_step.shape[0], 1), dtype=torch.float32, device=device)

        # threshold height to get occupancy and free labels
        thresh = occupancy_height_thresh
        y = local3D_step[:,1]
        occ_lbl[y>thresh,:] = 1
        occ_lbl[y<=thresh,:] = 2

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)
        map_coords = map_coords.to(device)

        ## Replicate label pooling
        grid = torch.empty(3, grid_dim[0], grid_dim[1], device=device)
        grid[:] = 1 / 3

        # If the robot does not project any values on the grid, then return the empty grid
        if map_coords.shape[0]==0:
            ego_grid_occ[k,:,:,:] = grid.unsqueeze(0)
            continue

        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        ego_grid_occ[k,:,:,:] = grid / grid.sum(dim=0)

    return ego_grid_occ



def ground_projection(points2D, local3D, sseg, sseg_labels, grid_dim, cell_size):
    ego_grid_sseg = torch.zeros((sseg.shape[0], sseg_labels, grid_dim[0], grid_dim[1]), dtype=torch.float32, device=sseg.device)

    for i in range(sseg.shape[0]): # sequence length
        sseg_step = sseg[i,:,:,:].unsqueeze(0) # 1 x 1 x H x W
        points2D_step = points2D[i]
        local3D_step = local3D[i]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        valid_inds = torch.nonzero(torch.where((z<3) & (z>0.5), 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        valid_inds = torch.nonzero(torch.where(y<1, 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)

        grid_sseg = label_pooling(sseg_step, points2D_step, map_coords, sseg_labels, grid_dim)
        grid_sseg = grid_sseg.unsqueeze(0)

        ego_grid_sseg[i,:,:,:] = grid_sseg

    return ego_grid_sseg


def label_pooling(sseg, points2D, map_coords, sseg_labels, grid_dim):
    # pool the semantic labels
    # For each bin get the frequencies of the class labels based on the labels projected
    # Each grid location will hold a probability distribution over the semantic labels
    grid = torch.ones((sseg_labels, grid_dim[0], grid_dim[1]), device='cuda')*(1/sseg_labels) # initially uniform distribution over the labels

    # If the robot does not project any values on the grid, then return the empty grid
    if map_coords.shape[0]==0:
        return grid
    pix_x, pix_y = points2D[:,0].long(), points2D[:,1].long()
    pix_lbl = sseg[0, 0, pix_y, pix_x]
    # SPEEDUP if map_coords is sorted, can switch to unique_consecutive
    uniq_rows = torch.unique(map_coords, dim=0)
    for i in range(uniq_rows.shape[0]):
        ucoord = uniq_rows[i,:]
        # indices of where ucoord can be found in map_coords
        ind = torch.nonzero(torch.where((map_coords==ucoord).all(axis=1), 1, 0)).squeeze(dim=1)
        bin_lbls = pix_lbl[ind]
        hist = torch.histc(bin_lbls, bins=sseg_labels, min=0, max=sseg_labels)
        hist = hist + 1e-5 # add a very small number to every location to avoid having 0s
        hist = hist / float(bin_lbls.shape[0])
        grid[:, ucoord[1], ucoord[0]] = hist
    return grid



def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    #map_coords = torch.zeros((len(x), 2), device='cuda')
    map_coords = torch.zeros((len(x), 2))
    xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
    zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
    xb = xb.int()
    zb = zb.int()
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    map_coords[map_coords>grid_dim[0]-1] = grid_dim[0]-1
    map_coords[map_coords<0] = 0
    return map_coords.long()



def get_gt_crops(abs_pose, pcloud, label_seq_all, agent_height, grid_dim, crop_size, cell_size):
    x_all, y_all, z_all = pcloud[0], pcloud[1], pcloud[2]
    episode_extend = abs_pose.shape[0]
    gt_grid_crops = torch.zeros((episode_extend, 1, crop_size[0], crop_size[1]), dtype=torch.int64)
    for k in range(episode_extend):
        # slice the gt map according to the agent height at every step
        x, y, label_seq = slice_scene(x_all.copy(), y_all.copy(), z_all.copy(), label_seq_all.copy(), agent_height[k])
        gt = get_gt_map(x, y, label_seq, abs_pose=abs_pose[k], grid_dim=grid_dim, cell_size=cell_size)
        _gt_crop = crop_grid(grid=gt.unsqueeze(0), crop_size=crop_size)
        gt_grid_crops[k,:,:,:] = _gt_crop.squeeze(0)
    return gt_grid_crops


def get_gt_map(x, y, label_seq, abs_pose, grid_dim, cell_size, color_pcloud=None, z=None):
    # Transform the ground-truth map to align with the agent's pose
    # The agent is at the center looking upwards
    point_map = np.array([x,y])
    angle = -abs_pose[2]
    rot_mat_abs = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    trans_mat_abs = np.array([[-abs_pose[1]],[abs_pose[0]]]) #### This is important, the first index is negative.
    ##rotating and translating point map points
    t_points = point_map - trans_mat_abs
    rot_points = np.matmul(rot_mat_abs,t_points)
    x_abs = torch.tensor(rot_points[0,:], device='cuda')
    y_abs = torch.tensor(rot_points[1,:], device='cuda')

    map_coords = discretize_coords(x=x_abs, z=y_abs, grid_dim=grid_dim, cell_size=cell_size)

    # Coordinates in map_coords need to be sorted based on their height, floor values go first
    # Still not perfect
    if z is not None:
        z = np.asarray(z)
        sort_inds = np.argsort(z)
        map_coords = map_coords[sort_inds,:]
        label_seq = label_seq[sort_inds,:]

    true_seg_grid = torch.zeros((grid_dim[0], grid_dim[1], 1), device='cuda')
    true_seg_grid[map_coords[:,1], map_coords[:,0]] = label_seq.clone()

    ### We need to flip the ground truth to align with the observations.
    ### Probably because the -y tp -z is a rotation about x axis which also flips the y coordinate for matteport.
    true_seg_grid = torch.flip(true_seg_grid, dims=[0])
    true_seg_grid = true_seg_grid.permute(2, 0, 1)

    true_occup_grid = true_seg_grid.detach().clone()
    true_occup_grid[(true_occup_grid!=0)&(true_occup_grid!=17)] = 1
    true_occup_grid[true_occup_grid==17] = 2

    if color_pcloud is not None:
        color_grid = torch.zeros((grid_dim[0], grid_dim[1], 3), device='cuda')
        color_grid[map_coords[:,1], map_coords[:,0],0] = color_pcloud[0]
        color_grid[map_coords[:,1], map_coords[:,0],1] = color_pcloud[1]
        color_grid[map_coords[:,1], map_coords[:,0],2] = color_pcloud[2]
        color_grid = torch.flip(color_grid, dims=[0])
        color_grid = color_grid.permute(2, 0 ,1)
        return true_seg_grid, true_occup_grid, color_grid/255.0
    else:
        return true_seg_grid, true_occup_grid


def crop_grid(grid, crop_size):
    # Assume input grid is already transformed such that agent is at the center looking upwards
    grid_dim_h, grid_dim_w = grid.shape[2], grid.shape[3]
    cx, cy = int(grid_dim_w/2.0), int(grid_dim_h/2.0)
    rx, ry = int(crop_size[0]/2.0), int(crop_size[1]/2.0)
    top, bottom, left, right = cx-rx, cx+rx, cy-ry, cy+ry
    return grid[:, :, top:bottom, left:right]

def slice_scene(x, y, z, label_seq, height, color_pcloud=None):
    # z = -z
    # Slice the scene below and above the agent
    below_thresh = height-0.2
    above_thresh = height+2.0
    all_inds = np.arange(y.shape[0])
    below_inds = np.where(z<below_thresh)[0]
    above_inds = np.where(z>above_thresh)[0]
    invalid_inds = np.concatenate( (below_inds, above_inds), 0) # remove the floor and ceiling inds from the local3D points
    inds = np.delete(all_inds, invalid_inds)
    x_fil = x[inds]
    y_fil = y[inds]
    z_fil = z[inds]
    label_seq_fil = torch.tensor(label_seq[inds], dtype=torch.float, device='cuda')
    if color_pcloud is not None:
        color_pcloud_fil = torch.tensor(color_pcloud[:,inds], dtype=torch.float, device='cuda')
        return x_fil, y_fil, z_fil, label_seq_fil, color_pcloud_fil
    else:
        return x_fil, y_fil, z_fil, label_seq_fil


def get_explored_grid(grid_sseg, thresh=0.5):
    # Use the ground-projected ego grid to get observed/unobserved grid
    # Single channel binary value indicating cell is observed
    # Input grid_sseg T x C x H x W (can be either H x W or cH x cW)
    # Returns T x 1 x H x W
    T, C, H, W = grid_sseg.shape
    grid_explored = torch.ones((T, 1, H, W), dtype=torch.float32).to(grid_sseg.device)
    grid_prob_max = torch.amax(grid_sseg, dim=1)
    inds = torch.nonzero(torch.where(grid_prob_max<=thresh, 1, 0))
    grid_explored[inds[:,0], 0, inds[:,1], inds[:,2]] = 0
    return grid_explored




import numpy as np
import quaternion
import torch
import os


def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint


def load_model(models, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    for model in models:
        if model in checkpoint['models']:
            models[model].load_state_dict(checkpoint['models'][model])
        else:
            raise Exception("Missing model in checkpoint: {}".format(model))
    return models


def get_2d_pose(position, rotation=None):
    # position is 3-element list
    # rotation is 4-element list representing a quaternion
    position = np.asarray(position, dtype=np.float32)
    
    if position.shape[-1] == 3:
        x = -position[2]
        y = -position[0]
        height = position[1]
    else:
        x = -position[1]
        y = -position[0]
        height = None

    if rotation is not None:
        rotation = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        axis = quaternion.as_euler_angles(rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        pose = x, y, o
    else:
        pose = x, y, 0.0
    return pose, height


def get_3d_pose(pose_2D, agent_pose_2D, agent_sim_pose, y_height, init_rot, cell_size):
    # Given a 2D grid (pose_2D) location, return its 3D abs pose in habitat sim coords
    init_rot = -init_rot
    dist_x = (pose_2D[0,0] - agent_pose_2D[0,0]) * cell_size
    dist_z = (pose_2D[0,1] - agent_pose_2D[0,1]) * cell_size
    init_rot_mat = torch.tensor([[torch.cos(init_rot), -torch.sin(init_rot)],[torch.sin(init_rot),torch.cos(init_rot)]], dtype=torch.float32)
    dist_vect = torch.tensor([dist_x,dist_z])
    dist_vect = dist_vect.reshape((2,1))
    rot_vect = torch.matmul(init_rot_mat,dist_vect)
    sim_pose_tmp_x, sim_pose_tmp_z = agent_sim_pose[0]-rot_vect[1], agent_sim_pose[1]-rot_vect[0]
    # revert changes from utils.get_sim_location
    sim_pose = np.zeros((3), dtype=np.float32)
    sim_pose[0] = -sim_pose_tmp_z
    sim_pose[1] = y_height
    sim_pose[2] = -sim_pose_tmp_x
    return sim_pose.tolist() 


def transform_to_map_coords(sg, position, abs_pose, grid_size, cell_size, device):
    pose, _ = get_2d_pose(position=position)
    agent_rel_pose = get_rel_pose(pos2=pose, pos1=abs_pose)
    agent_rel_pose = torch.Tensor(agent_rel_pose).unsqueeze(0).float()
    agent_rel_pose = agent_rel_pose.to(device)
    _pose_coords = get_coord_pose(sg, agent_rel_pose, abs_pose, grid_size, cell_size, device) # B x T x 3

    visible_position = 1
    # if goal pose coords is 0,0 then goal is outside the current map. Use an empty heatmap
    if _pose_coords[0,0,0]==0 and _pose_coords[0,0,1]==0:
        _pose_coords = torch.tensor([[[-200,-200]]])
        visible_position = 0
    return _pose_coords, visible_position


def transform_ego_to_geo(ego_point, pose_coords, abs_pose_coords, abs_poses, t):
    # ego_point is point to transform
    # pose_coords is agent's ego centric pose (always in the center of the map)
    # abs_pose_coords is agent's pose with respect to first pose in the episode
    rel_rot = torch.tensor(abs_poses[0][2]) - torch.tensor(abs_poses[t][2])
    dist_x = (ego_point[0,0,0] - pose_coords[0,0,0])
    dist_z = (ego_point[0,0,1] - pose_coords[0,0,1])
    rel_rot_mat = torch.tensor([[torch.cos(rel_rot), -torch.sin(rel_rot)],[torch.sin(rel_rot),torch.cos(rel_rot)]], dtype=torch.float32)
    dist_vect = torch.tensor([dist_x,dist_z])
    dist_vect = dist_vect.reshape((2,1))
    rot_vect = torch.matmul(rel_rot_mat,dist_vect)

    abs_coords_x = abs_pose_coords[0,0,0] + rot_vect[0]
    abs_coords_z = abs_pose_coords[0,0,1] + rot_vect[1]
    abs_coords = torch.tensor([[[abs_coords_x, abs_coords_z]]])
    return abs_coords


def get_coord_pose(sg, rel_pose, init_pose, grid_dim, cell_size, device=None):
    # Create a grid where the starting location is always at the center looking upwards (like the ground-projected grids)
    # Then use the spatial transformer to move that location at the right place
    if isinstance(init_pose, list) or isinstance(init_pose, tuple):
        init_pose = torch.tensor(init_pose).unsqueeze(0)
    else:
        init_pose = init_pose.unsqueeze(0)

    zero_pose = torch.tensor([[0., 0., 0.]])
    if device!=None:
        init_pose = init_pose.to(device)
        zero_pose = zero_pose.to(device)

    zero_coords = discretize_coords(x=zero_pose[:,0],
                                            z=zero_pose[:,1],
                                            grid_dim=(grid_dim, grid_dim),
                                            cell_size=cell_size)

    pose_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=torch.float32)#.to(device)
    pose_grid[0,0,zero_coords[0,0], zero_coords[0,1]] = 1

    pose_grid_transf = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose.unsqueeze(0))
    
    pose_grid_transf = pose_grid_transf.squeeze(0).squeeze(0)
    inds = unravel_index(pose_grid_transf.argmax(), pose_grid_transf.shape)

    pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64)#.to(device)
    pose_coord[0,0,0] = inds[1] # inds is y,x
    pose_coord[0,0,1] = inds[0]
    return pose_coord



def decide_stop_vln(pred_goal_dist, stop_dist, ltg_cons=True):
    if pred_goal_dist <= stop_dist and ltg_cons:
        return True
    else:
        return False



# Return success, SPL, soft_SPL, distance_to_goal measures
def get_metrics_vln(sim,
                goal_position,
                success_distance,
                start_end_episode_distance,
                agent_episode_distance,
                sim_agent_poses,
                stop_signal):

    curr_pos = sim.get_agent_state().position
    # returns distance to the closest goal position
    distance_to_goal = sim.geodesic_distance(curr_pos, goal_position)

    if distance_to_goal <= success_distance and stop_signal:
        success = 1.0
    else:
        success = 0.0

    spl = success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    ep_soft_success = max(0, (1 - distance_to_goal / start_end_episode_distance) )
    soft_spl = ep_soft_success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    # Navigation error is the min(geodesic_distance(agent_pos, goal)) over all points in the agent path
    min_dist=99999
    for i in range(len(sim_agent_poses)):
        dist = sim.geodesic_distance(sim_agent_poses[i], goal_position)
        if dist < min_dist:
            min_dist = dist

    # If at any point in the path we were within success distance then oracle success=1
    if min_dist <= success_distance:
        oracle_success = 1.0
    else:
        oracle_success = 0.0

    metrics = {'distance_to_goal':distance_to_goal,
               'success':success,
               'spl':spl,
               'softspl':soft_spl,
               'trajectory_length':agent_episode_distance,
               'navigation_error':min_dist,
               'oracle_success': oracle_success}
    return metrics





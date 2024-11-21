
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image

'''
MP3D original semantic labels and reduced set correspondence
# Original set from here: https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
0 void 0
1 wall 15 structure
2 floor 17 free-space
3 chair 1
4 door 2
5 table 3
6 picture 18
7 cabinet 19
8 cushion 4
9 window 15 structure
10 sofa 5
11 bed 6
12 curtain 16 other
13 chest_of_drawers 20
14 plant 7
15 sink 8
16 stairs 17 free-space
17 ceiling 17 free-space
18 toilet 9
19 stool 21
20 towel 22
21 mirror 16 other
22 tv_monitor 10
23 shower 11
24 column 15 structure
25 bathtub 12
26 counter 13
27 fireplace 23
28 lighting 16 other
29 beam 16 other
30 railing 16 other
31 shelving 16 other
32 blinds 16 other
33 gym_equipment 24
34 seating 25
35 board_panel 16 other
36 furniture 16 other
37 appliances 14
38 clothes 26
39 objects 16 other
40 misc 16 other
'''
# 27 categories which include the 21 object categories in the habitat challenge
label_conversion_40_27 = {-1:0, 0:0, 1:15, 2:17, 3:1, 4:2, 5:3, 6:18, 7:19, 8:4, 9:15, 10:5, 11:6, 12:16, 13:20, 14:7, 15:8, 16:17, 17:17,
                    18:9, 19:21, 20:22, 21:16, 22:10, 23:11, 24:15, 25:12, 26:13, 27:23, 28:16, 29:16, 30:16, 31:16, 32:16,
                    33:24, 34:25, 35:16, 36:16, 37:14, 38:26, 39:16, 40:16}
color_mapping_27 = {
    0:(255,255,255), # white
    1:(128,128,0), # olive (dark yellow)
    2:(0,0,255), # blue
    3:(255,0,0), # red
    4:(255,0,255), # magenta
    5:(0,255,255), # cyan
    6:(255,165,0), # orange
    7:(255,255,0), # yellow
    8:(128,128,128), # gray
    9:(128,0,0), # maroon
    10:(255,20,147), # pink 
    11:(0,128,0), # dark green
    12:(128,0,128), # purple
    13:(0,128,128), # teal
    14:(0,0,128), # navy (dark blue)
    15:(210,105,30), # chocolate
    16:(188,143,143), # rosy brown
    17:(0,255,0), # green
    18:(255,215,0), # gold
    19:(0,0,0), # black
    20:(192,192,192), # silver
    21:(138,43,226), # blue violet
    22:(255,127,80), # coral
    23:(238,130,238), # violet
    24:(245,245,220), # beige
    25:(139,69,19), # saddle brown
    26:(64,224,208) # turquoise
}

# three label classification (0:void, 1:occupied, 2:free)
label_conversion_40_3 = {-1:0, 0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2,
                    18:1, 19:1, 20:1, 21:1, 22:1, 23:1, 24:1, 25:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1,
                    33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1, 40:1}
color_mapping_3 = {
    0:(255,255,255), # white
    1:(0,0,255), # blue
    2:(0,255,0), # green
}

# visualize a scene map with it's sampled waypoints from RxR
def vis_episode(gt_map_semantic, pose_coords, name="map_tmp", color_mapping=27):
    color_map = colorize_grid(gt_map_semantic.unsqueeze(0).unsqueeze(0), color_mapping=color_mapping)
    im_color_map = color_map[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(10 ,8))
    plt.axis('off')
    plt.imshow(im_color_map)
    for i in range(len(pose_coords)):
        point = pose_coords[i]
        if point[0]>=0 and point[1]>=0:
            plt.scatter(point[0], point[1], color="blue", s=50)
    plt.savefig(name+'.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def vis_heatmaps(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    # heatmaps are 1 x h x w
    arr = [pred, gt]
    n = len(arr)
    plt.figure(figsize=(10,5))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, n, i+1)
        ax.axis('off')
        plt.imshow(data)
    plt.show()



def write_img(img, savepath, name):
    # img: T x 3 x dim x dim, assumed normalized
    for i in range(img.shape[0]):
        vis_img = img[i,:,:,:].cpu().numpy()
        vis_img = np.transpose(vis_img, (1,2,0))
        im_path = savepath + str(i) + "_" + name + ".png"
        cv2.imwrite(im_path, vis_img[:,:,::-1]*255.0)



def colorize_grid(grid, color_mapping=27): # to pass into tensorboardX video
    # Input: grid -- B x T x C x grid_dim x grid_dim, where C=1,T=1 when gt and C=41,T>=1 for other
    # Output: grid_img -- B x T x 3 x grid_dim x grid_dim
    grid = grid.detach().cpu().numpy()
    grid_img = np.zeros((grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4], 3),  dtype=np.uint8)
    if grid.shape[2] > 1:
        # For cells where prob distribution is all zeroes (or uniform), argmax returns arbitrary number (can be true for the accumulated maps)
        grid_prob_max = np.amax(grid, axis=2)
        inds = np.asarray(grid_prob_max<=0.05).nonzero() # if no label has prob higher than k then assume unobserved
        grid[inds[0], inds[1], 0, inds[2], inds[3]] = 1 # assign label 0 (void) to be the dominant label
        grid = np.argmax(grid, axis=2) # B x T x grid_dim x grid_dim
    else:
        grid = grid.squeeze(2)

    if color_mapping==27:
        color_mapping = color_mapping_27
    else:
        color_mapping = color_mapping_3
    for label in color_mapping.keys():
        grid_img[ grid==label ] = color_mapping[label]
    
    return torch.tensor(grid_img.transpose(0, 1, 4, 2, 3), dtype=torch.uint8)


def write_tensor_image(grid, savepath, name, sseg_labels=27):
    # grid: T x C x dim x dim 
    grid_imgs = colorize_grid(grid.unsqueeze(0), color_mapping=sseg_labels)
    grid_imgs = grid_imgs.squeeze(0)
    grid_imgs = grid_imgs.detach().cpu().numpy()
    for t in range(grid_imgs.shape[0]):
        im = grid_imgs[t,:,:,:].transpose(1,2,0)
        im_path = savepath + str(t) + "_" + name + ".png"
        cv2.imwrite(im_path, im[:,:,::-1])


def write_tensor_imgSegm(img, savepath, name, t=None, labels=27, waypoints=None):
    # pred: T x C x dim x dim
    if img.shape[1] > 1:
        img = torch.argmax(img.cpu(), dim=1, keepdim=True) # T x 1 x cH x cW
    img_labels = img.squeeze(1)

    for i in range(img_labels.shape[0]):
        img0 = img_labels[i,:,:]

        vis_img = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)
        
        if labels==27:
            color_mapping = color_mapping_27
        else:
            color_mapping = color_mapping_3

        for label in color_mapping.keys():
            vis_img[ img0==label ] = color_mapping[label]
        
        if t is None:
            im_path = savepath + str(i) + "_" + name + ".png"
        else:
            im_path = savepath + name + "_" + str(t) + "_" + str(i) + ".png"

        if waypoints != None:
            for coords in waypoints:
                vis_img[coords[1]-3:coords[1]+3,coords[0]-3:coords[0]+3,:] = np.array([0,0,1])

        cv2.imwrite(im_path, vis_img[:,:,::-1])


def display_sample(rgb_obs, depth_obs, t, sseg_img=None, savepath=None):
    # sseg_img is semantic observation from Matterport habitat
    depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
        
    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(rgb_img)
    plt.savefig(savepath+str(t)+"_rgb.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()

    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(depth_img)
    plt.savefig(savepath+str(t)+"_depth.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()


def save_map_goal(gt_map_semantic, pose_coords, goal_pose_coords, save_img_dir_, t):
    color_map_sem = colorize_grid(gt_map_semantic)
    im = color_map_sem[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(10 ,7))
    plt.axis('off')
    plt.imshow(im)
    if goal_pose_coords[0,0,0]>=0 and goal_pose_coords[0,0,1]>=0:
        plt.scatter(goal_pose_coords[0,0,0], goal_pose_coords[0,0,1], color="magenta", s=70)
    plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="blue", s=70)
    plt.savefig(save_img_dir_+str(t)+'.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()



def save_map_pred_steps(spatial_in, spatial_pred, objects_pred, ego_img_segm, save_img_dir_, t):
    
    color_spatial_in = colorize_grid(spatial_in, color_mapping=3)
    im_spatial_in = color_spatial_in[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_spatial_pred = colorize_grid(spatial_pred, color_mapping=3)
    im_spatial_pred = color_spatial_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_objects_pred = colorize_grid(objects_pred, color_mapping=27)
    im_objects_pred = color_objects_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_ego_img_segm = colorize_grid(ego_img_segm, color_mapping=27)
    im_ego_img_segm = color_ego_img_segm[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(im_spatial_in)
    plt.savefig(save_img_dir_+str(t)+"_im_spatial_in.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()

    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(im_spatial_pred)
    plt.savefig(save_img_dir_+str(t)+"_im_spatial_pred.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()    

    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(im_objects_pred)
    plt.savefig(save_img_dir_+str(t)+"_im_objects_pred.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()

    plt.figure(figsize=(12 ,8))
    plt.axis('off')
    plt.imshow(im_ego_img_segm)
    plt.savefig(save_img_dir_+str(t)+"_im_ego_img_segm.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
    plt.close()    


def show_waypoint_pred(map_semantic, savepath, num_points, ltg=None, pose_coords=None, pred_waypoints=None, gt_waypoints=None):
    # Waypoints are provided in map pose coordinates
    color_map = colorize_grid(map_semantic.unsqueeze(0).unsqueeze(0))
    im_color_map = color_map[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(10 ,8))
    plt.axis('off')
    plt.imshow(im_color_map)

    for i in range(num_points):
        if gt_waypoints is not None:
            point_gt = gt_waypoints[i]
            if point_gt[0]>=0 and point_gt[1]>=0:
                plt.scatter(point_gt[0], point_gt[1], color="blue", s=50)
                plt.text(point_gt[0], point_gt[1], s=str(i), color="blue")
        if pred_waypoints is not None:
            point_pred = pred_waypoints[i]
            if point_pred[0]>=0 and point_pred[1]>=0:
                plt.scatter(point_pred[0], point_pred[1], color="red", s=50)
                plt.text(point_pred[0], point_pred[1], s=str(i), color="red")
    # ltg and agent position
    if ltg is not None:
        plt.scatter(ltg[0,0,0], ltg[0,0,1], color="magenta", s=50)
    if pose_coords is not None:
        plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="green", s=50)

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
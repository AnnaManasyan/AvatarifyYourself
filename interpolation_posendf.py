### Paste this file in the root repo of PoseNDF and change the paths accordingly
import argparse
from configs.config import load_config
# General config

#from model_quat import  train_manifold2 as train_manifold
from model.posendf import PoseNDF
import shutil
from data.data_splits import amass_splits
# import ipdb
import torch
import numpy as np
# from body_model import BodyModel
# from exp_utils import renderer

from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
from tqdm import tqdm
from pytorch3d.io import save_obj
import os

from torch.autograd import grad

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad

opt = load_config('/home/hiwi-1/Documents/PoseNDF/configs/config.yaml')
net = PoseNDF(opt)
device= 'cuda:0'
ckpt = torch.load('pretrained_models/checkpoint_epoch_best.tar', map_location='cpu')['model_state_dict']
net.load_state_dict(ckpt)
net.eval()
net = net.to(device)
batch_size= 10
#if noisy pose path not given, then generate random quaternions
# noisy_pose = torch.rand((batch_size,21,4))
# noisy_pose = torch.nn.functional.normalize(noisy_pose,dim=2).to(device=device)
# noisy_pose.requires_grad = True
# for it in range(100):
#     net_pred = net(noisy_pose, train=False)
#     print(torch.mean(net_pred['dist_pred']))
#     grad_out = gradient(noisy_pose, net_pred['dist_pred']).reshape(-1, 84)
#     noisy_pose = noisy_pose - (net_pred['dist_pred']*grad_out).reshape(-1, 21,4)


#### YOU CAN USE ANY AMASS ANIMATION SEQUENCE HERE
full_seq = np.load('motions/jacks.npz')

## Get starting and end pose and translation
## SAMPLE 2 RANDOM POSES, I CHOSE 44, and 104 as it was end and start of
## jumping jacks
start_pose = full_seq['poses'][44, 3:66].reshape(21,3) ## The frame 44 was determined by interactive frames in 'amass_vis.py' script
end_pose = full_seq['poses'][104, 3:66].reshape(21,3)

start_trans = full_seq['trans'][44]
end_trans = full_seq['trans'][104]

## Total t steps
total_steps = 69.0 ## Total steps will be +1 of this
step_size = 1.0 / total_steps
t_list = np.arange(0,1+step_size,step_size)

### Declaring structures to save the output
output_pose = np.zeros((len(t_list), 22, 3))
output_trans = np.zeros((len(t_list), 3))

### Convert poses to quaternion as that is what poseNDF functions with
start_pose_q = axis_angle_to_quaternion(torch.from_numpy(start_pose))
end_pose_q = axis_angle_to_quaternion(torch.from_numpy(end_pose))
start_root_pose = axis_angle_to_quaternion(torch.from_numpy(full_seq['poses'][44, :3]))
end_root_pose = axis_angle_to_quaternion(torch.from_numpy(full_seq['poses'][104, :3]))

### We now interpolate all the poses and give them as a single batch
interpolated_poses = torch.zeros((len(t_list), 21, 4))

idx = 0
for t in t_list:
    curr_pose = t*end_pose_q + (1-t)*start_pose_q
    row_norms = torch.linalg.norm(curr_pose, axis=1, keepdims=True)
    curr_pose = curr_pose / row_norms

    curr_trans = t*end_trans + (1-t)*start_trans
    
    interpolated_poses[idx] =  curr_pose
    output_trans[idx] = curr_trans
    ## The root orientation is directly copied
    root_pose = t*end_root_pose + (1-t)*start_root_pose
    root_pose = root_pose / torch.linalg.norm(root_pose)

    output_pose[idx, 0, :] = quaternion_to_axis_angle(root_pose).cpu().numpy()
    idx +=1

interpolated_poses = interpolated_poses.to(device)
interpolated_poses.requires_grad = True

dist = 1
idx = 0
for it in range(100):
    net_pred = net(interpolated_poses, train=False)
    dist = torch.mean(net_pred['dist_pred'])
    # if idx % 50 == 0:
    print("Printing for idx: ", idx, "dist: ", dist)
    grad_out = gradient(interpolated_poses, net_pred['dist_pred']).reshape(-1, 84)
    interpolated_poses = interpolated_poses - (net_pred['dist_pred']*grad_out).reshape(-1, 21,4)
    idx += 1

interpolated_poses = quaternion_to_axis_angle(interpolated_poses)

interpolated_poses_np = interpolated_poses.detach().cpu().numpy()

output_pose[:, 1: , :] = interpolated_poses_np

output_trans = output_trans.reshape((len(t_list), -1))
output_pose = output_pose.reshape((len(t_list), -1))

output_dict = {'trans': output_trans, 'poses': output_pose, 'betas': full_seq['betas'],
               'gender': full_seq['gender']}

np.savez('posendf_interpolate_joint_test.npz', **output_dict)
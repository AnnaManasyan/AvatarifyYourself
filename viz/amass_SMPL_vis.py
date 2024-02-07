import numpy as np
import torch
from os.path import join
import sys


sys.path.append('./')
from lib.th_SMPL import th_batch_SMPL
from lib.rotation_conversions import *
from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from psbody.mesh.lines import Lines
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
import time
from copy import deepcopy

import time

from pynput import keyboard


##### FOR MAKING MESH FACE CAMERA
theta_z = np.radians(90)
theta_y = np.radians(90)
theta_x = np.radians(90)
rot_mat_z = np.array([[np.cos(theta_z), np.sin(theta_z) ,0 ], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
rot_mat_y = np.array([[np.cos(theta_y), 0 ,np.sin(theta_y) ], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
rot_mat_x = np.array([ [1,0,0], [0, np.cos(theta_x), np.sin(theta_x)], [0, -np.sin(theta_x), np.cos(theta_x)]])
# rot_mat_final = rot_mat_y @ rot_mat_z
rot_mat_final = np.eye(3) @ rot_mat_y @ rot_mat_x @ rot_mat_z

#### FOR DETECTING USER INPUT
space_pressed = False #Global variable which is set to true if space is pressed
def on_space_press(key):
    if key == keyboard.Key.space:
        global space_pressed 
        space_pressed = True

listener = keyboard.Listener(
    on_press=on_space_press)
listener.start()

comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### We read the file with the SMPL-X params ###
amass_npz_fname = 'interpolated_poses/posendf_interpolate_joint_test.npz' # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

### Loading SMPL-X body model
smpl_male = th_batch_SMPL('neutral', False)

num_betas = 10 # number of body parameters

### We sample the needed joints from the SMPL-X model to convert to SMPL model
poses = torch.FloatTensor(bdata['poses']).reshape(-1, 55, 3)
pose = poses[:, np.arange(0, 23).tolist() + [37], :].reshape(-1, 72)
trans = torch.from_numpy(bdata['trans']).float()
betas = torch.from_numpy(bdata['betas'][:10]).float()


time_length = len(bdata['trans'])

## We call new smpl model with SMPL pose and translations
verts, J = smpl_male(pose=pose, trans=trans, betas=betas, scale=1)[:2]
verts = verts.numpy()

# imw, imh=1600, 1600
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
mv = MeshViewers([1, 1])
color = 'white'

interaction_frames = []
for i in range(verts.shape[0]):
    meshes = []
    tmp_verts = (rot_mat_final @ verts[i].T).T
    meshes += [Mesh(tmp_verts, smpl_male.faces.numpy(), vc=color)]# this will be your scene_mesh
    mv[0][0].static_meshes = meshes
    if space_pressed:
        interaction_frame = i+1
        interaction_frames += [interaction_frame]
        space_pressed = False
    time.sleep(0.01)

print("Interaction Start At Frame: ", interaction_frames)
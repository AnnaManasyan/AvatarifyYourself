from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
import torch
import numpy as np 
import matplotlib.pyplot as plt
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp
from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
import time

comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### We read the file with the SMPL params ###
amass_npz_fname = 'interpolated_poses/posendf_interpolate_joint_test.npz' # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = "neutral"

### Loading SMPL-X body model
bm_fname = osp.join('body_models', 'smpl/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters
poses = bdata['poses']
trans = bdata['trans']
betas = bdata['betas']
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, model_type='smpl').to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
}

body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'trans']})


mv = MeshViewers([1, 1])
color = 'white'

for i in range(time_length):
    # body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    # mv.set_static_meshes([body_mesh])
    # body_image = mv.render(render_wireframe=False)
    # show_image(body_image)
    meshes = []
    meshes += [Mesh(c2c(body_pose_beta.v[i]), faces, vc=color)]
    mv[0][0].static_meshes = meshes
    # if i == 0:
    #     mv[0][1].static_meshes = meshes
    
    time.sleep(0.03)
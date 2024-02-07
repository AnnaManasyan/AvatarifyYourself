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

smpl_male = th_batch_SMPL('male', False)
# smpl_female = th_batch_SMPL('female', False)


mv = MeshViewers([1, 1])
color = 'white'

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', "--data_path", default='data')
    args = parser.parse_args()

    # load smpl params
    obj_name = 'backpack'
    obj_params = np.load(join(args.data_path, 'object_fit_all.npz'), allow_pickle=True)
    smpl_params = np.load(join(args.data_path, 'smpl_fit_all.npz'), allow_pickle=True)
    betas = torch.from_numpy(smpl_params['betas'].mean(0)).float()

    # offset for transforming global (you won't need)
    J_offset = smpl_male(pose=torch.zeros((1, 72)), trans=torch.zeros((1, 3)), betas=betas, scale=1)[1][0][0].numpy()

    pose = torch.FloatTensor(smpl_params['poses']).reshape(-1, 52, 3) # loaded all body pose of 52 joints (24 in your case)
    pose = pose[:, np.arange(0, 23).tolist() + [37]].reshape(-1, 72) # only select hand joints (you won't need)
    trans = torch.from_numpy(smpl_params['trans']).float()

    # applying global transformation (you won't need)
    delta = axis_angle_to_matrix(torch.FloatTensor([[-np.pi, 0, 0]])).repeat(pose.shape[0], 1, 1)
    pose[:, :3] = matrix_to_axis_angle(torch.bmm(delta, axis_angle_to_matrix(deepcopy(pose[:, :3]))))
    trans = torch.matmul(delta[0], trans.T + J_offset[:, np.newaxis]).T - J_offset[np.newaxis]

    # transform smpl
    verts, J = smpl_male(pose=pose, trans=trans, betas=betas, scale=1)[:2]
    verts = verts.numpy()

    # ensuring foot stays on the floor (
    heights = deepcopy(verts[:, :, 1].min(1)[:, np.newaxis])
    verts[:, :, 1] -= heights
    J = J.numpy()
    J[:, :, 1] -= heights

    # loading obj
    obj_mesh = Mesh()
    obj_mesh.load_from_file(join(args.data_path, obj_name, obj_name + '_f1000.ply')) # replace this with your scene.ply


    #### the next chunk you won't need, since you only visualse static scenes.
    obj_mesh_centre = obj_mesh.v.mean(0)
    obj_mesh.v -= obj_mesh_centre
    # loading obj params
    angle = torch.from_numpy(obj_params['angles']).float()
    trans = torch.from_numpy(obj_params['trans']).float()
    angle = matrix_to_axis_angle(torch.matmul(delta, axis_angle_to_matrix(deepcopy(angle)))).numpy()
    trans = torch.matmul(delta[0].T, trans.T).T.numpy()
    # transform obj
    obj_verts = np.einsum('ilk,jk->ijl', axis_angle_to_matrix(torch.from_numpy(angle).float()).numpy(),
                          obj_mesh.v) + trans[:, np.newaxis]
    obj_verts[:, :, 1] -= heights

    # visualise
    for i in range(verts.shape[0]):
        meshes = []
        meshes += [Mesh(verts[i], smpl_male.faces.numpy(), vc=color)]
        meshes += [Mesh(obj_verts[i], obj_mesh.f)] # this will be your scene_mesh
        mv[0][0].static_meshes = meshes
        time.sleep(0.02)

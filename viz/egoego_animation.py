import numpy as np
import time
from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c

theta_z = np.radians(180)
theta_y = np.radians(90)
theta_x = np.radians(180)
rot_mat_z = np.array([[np.cos(theta_z), np.sin(theta_z) ,0 ], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
rot_mat_y = np.array([[np.cos(theta_y), 0 ,np.sin(theta_y) ], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
rot_mat_x = np.array([ [1,0,0], [0, np.cos(theta_x), np.sin(theta_x)], [0, -np.sin(theta_x), np.cos(theta_x)]])
# rot_mat_final = rot_mat_y @ rot_mat_z
rot_mat_final = np.eye(3) @ rot_mat_y @ rot_mat_x @ rot_mat_z


mesh_file_1 = np.load("egoego/outmesh_gimo_gt.npz")
mesh_file_2 = np.load("egoego/outmesh_gimo.npz")
# motion_file = np.load("smpl_seq1.npz")
# translation = motion_file['trans']
# poses = motion_file['poses']

vertices_1 = mesh_file_1['v_seq']
faces_1 = mesh_file_1['f']
vertices_2 = mesh_file_2['v_seq']
faces_2 = mesh_file_2['f']
rot_vertices_1 = np.zeros_like(vertices_1)
rot_vertices_2 = np.zeros_like(vertices_2)
for i in range(vertices_1.shape[0]):
    rot_vertices_1[i] = (rot_mat_final @ vertices_1[i].T).T
    rot_vertices_2[i] = (rot_mat_final @ vertices_2[i].T).T


mv = MeshViewers([1, 1])
color_1 = 'green'
color_2= 'red'

## Now we visualize the motion
for i in range(vertices_1.shape[0]):
    meshes = []
    meshes += [Mesh(c2c(rot_vertices_1[i]), faces_1, vc=color_1)]
    # meshes += [Mesh(c2c(rot_vertices_2[i]), faces_2, vc=color_2)]
    mv[0][0].static_meshes = meshes
    time.sleep(0.1)
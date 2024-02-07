import numpy as np
import time
from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c

theta_z = np.radians(0)
theta_y = np.radians(90)
theta_x = np.radians(90)
rot_mat_z = np.array([[np.cos(theta_z), np.sin(theta_z) ,0 ], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
rot_mat_y = np.array([[np.cos(theta_y), 0 ,np.sin(theta_y) ], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
rot_mat_x = np.array([ [1,0,0], [0, np.cos(theta_x), np.sin(theta_x)], [0, -np.sin(theta_x), np.cos(theta_x)]])
# rot_mat_final = rot_mat_y @ rot_mat_z
rot_mat_final = np.eye(3) @ rot_mat_y @ rot_mat_x @ rot_mat_z


mesh_file = np.load("egoego/outmesh_gt.npz")
# motion_file = np.load("smpl_seq1.npz")
# translation = motion_file['trans']
# poses = motion_file['poses']

vertices = mesh_file['v_seq']
vertex = mesh_file['v_seq'][0]
faces = mesh_file['f']
rot_vertices = np.zeros_like(vertices)
for i in range(vertices.shape[0]):
    rot_vertices[i] = (rot_mat_final @ vertices[i].T).T

mv = MeshViewers([1, 1])
color = 'white'

## Now we visualize the motion
for i in range(vertices.shape[0]):
    meshes = []
    meshes += [Mesh(c2c(rot_vertices[i]), faces, vc=color)]
    mv[0][0].static_meshes = meshes
    if i==0:
        time.sleep(1)
    time.sleep(0.3)
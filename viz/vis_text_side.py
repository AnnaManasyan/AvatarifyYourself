import trimesh
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy as np
import time

new_mesh_agniv = Mesh()
new_mesh_agniv.load_from_obj(filename='agniv_mask/mesh.obj')
new_mesh_agniv.texture_filepath = 'agniv_mask/albedo.png'

new_mesh_vera = Mesh()
new_mesh_vera.load_from_obj(filename='vera_mask/mesh.obj')
new_mesh_vera.texture_filepath = 'vera_mask/albedo.png'

new_mesh_anna = Mesh()
new_mesh_anna.load_from_obj(filename='anna_mask/mesh.obj')
new_mesh_anna.texture_filepath = 'anna_mask/albedo.png'

theta_z = np.radians(90)
theta_y = np.radians(90)
theta_x = np.radians(90)
rot_mat_z = np.array([[np.cos(theta_z), np.sin(theta_z) ,0 ], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
rot_mat_y = np.array([[np.cos(theta_y), 0 ,np.sin(theta_y) ], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
rot_mat_x = np.array([ [1,0,0], [0, np.cos(theta_x), np.sin(theta_x)], [0, -np.sin(theta_x), np.cos(theta_x)]])
rot_mat_final = rot_mat_y @ rot_mat_z
rot_mat_final = np.eye(3) @ rot_mat_y @ rot_mat_x @ rot_mat_z


mesh_file_agniv = np.load("seq_animation/agniv_motion.npz")
mesh_file_vera = np.load("seq_animation/vera_motion.npz")
mesh_file_anna = np.load("seq_animation/anna_motion.npz")

faces_agniv = mesh_file_agniv['f']
faces_vera = mesh_file_vera['f']
faces_anna = mesh_file_anna['f']

vertices_agniv = mesh_file_agniv['v_seq']
vertices_anna = mesh_file_anna['v_seq']
vertices_vera = mesh_file_vera['v_seq']

rot_vertices_agniv = np.zeros_like(vertices_agniv)
for i in range(mesh_file_agniv['v_seq'].shape[0]):
    rot_vertices_agniv[i] = (rot_mat_final @ vertices_agniv[i].T).T

rot_vertices_anna = np.zeros_like(vertices_anna)
for i in range(mesh_file_agniv['v_seq'].shape[0]):
    rot_vertices_anna[i] = (rot_mat_final @ vertices_anna[i].T).T

rot_vertices_vera = np.zeros_like(vertices_vera)
for i in range(mesh_file_agniv['v_seq'].shape[0]):
    rot_vertices_vera[i] = (rot_mat_final @ vertices_vera[i].T).T

mv = MeshViewers([1, 3])
color = 'white'


for i in range(vertices_agniv.shape[0]):
    meshes_1 = []
    meshes_2 = []
    meshes_3 = []
    new_mesh_agniv.v = c2c(rot_vertices_agniv[i])
    new_mesh_agniv.f = faces_agniv

    new_mesh_anna.v = c2c(rot_vertices_anna[i])
    new_mesh_anna.f = faces_anna

    new_mesh_vera.v = c2c(rot_vertices_vera[i])
    new_mesh_vera.f = faces_vera
    # new_mesh_1 = Mesh(c2c(rot_vertices[0]), faces, vc=color)
    # new_mesh.materials_filepath = 'front_img_mesh/material.mtl'
    # new_mesh.materials_file = open(new_mesh.materials_filepath, 'r').readlines()
    # new_mesh_1.texture_filepath = 'front_img_mesh/albedo.png'
    meshes_1 += [new_mesh_agniv]
    meshes_2 += [new_mesh_anna]
    meshes_3 += [new_mesh_vera]

    mv[0][0].static_meshes = meshes_1
    mv[0][1].static_meshes = meshes_2
    mv[0][2].static_meshes = meshes_3
    time.sleep(0.001)
import trimesh
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy as np
import time
new_mesh = Mesh()
new_mesh.load_from_obj(filename='vera_text/mesh.obj')
new_mesh.texture_filepath = 'vera_text/albedo.png'

mv = MeshViewers([1, 1])
color = 'white'

meshes = []
meshes += [new_mesh]
mv[0][0].static_meshes = meshes

# theta_z = np.radians(270)
# theta_y = np.radians(90)
# theta_x = np.radians(90)
# rot_mat_z = np.array([[np.cos(theta_z), np.sin(theta_z) ,0 ], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
# rot_mat_y = np.array([[np.cos(theta_y), 0 ,np.sin(theta_y) ], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
# rot_mat_x = np.array([ [1,0,0], [0, np.cos(theta_x), np.sin(theta_x)], [0, -np.sin(theta_x), np.cos(theta_x)]])
# rot_mat_final = rot_mat_y @ rot_mat_z
# rot_mat_final = np.eye(3) @ rot_mat_y @ rot_mat_x @ rot_mat_z


# mesh_file = np.load("front_img_motion.npz")

# vertices = mesh_file['v_seq']
# vertex = mesh_file['v_seq'][0]
# faces = mesh_file['f']
# rot_vertices = np.zeros_like(vertices)
# for i in range(vertices.shape[0]):
#     rot_vertices[i] = (rot_mat_final @ vertices[i].T).T


# for i in range(vertices.shape[0]):
#     meshes_1 = []
#     new_mesh.v = c2c(rot_vertices[i])
#     new_mesh.f = faces
#     # new_mesh_1 = Mesh(c2c(rot_vertices[0]), faces, vc=color)
#     # new_mesh.materials_filepath = 'front_img_mesh/material.mtl'
#     # new_mesh.materials_file = open(new_mesh.materials_filepath, 'r').readlines()
#     # new_mesh_1.texture_filepath = 'front_img_mesh/albedo.png'
#     meshes_1 += [new_mesh]
#     mv[0][0].static_meshes = meshes_1
#     time.sleep(0.01)
    



# import trimesh
# import pyrender
# # fuze_trimesh = trimesh.load('dump_folder/mesh.obj')
# # mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=rot_vertices[0], faces=faces))

# # scene = pyrender.Scene()
# # scene.add(mesh)
# # pyrender.Viewer(scene, use_raymond_lighting=True)

# scene = pyrender.Scene()
# for i in range(vertices.shape[0]):
#     mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=rot_vertices[i], faces=faces))
#     node1 = scene.add(mesh)
#     pyrender.Viewer(scene, use_raymond_lighting=True)

# mv = MeshViewers([1, 2])
# color = 'white'

# meshes = []
# meshes += [mesh]
# mv[0][0].static_meshes = meshes
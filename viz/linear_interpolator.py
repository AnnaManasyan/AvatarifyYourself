import numpy as np
from scipy.spatial.transform import Rotation

full_seq = np.load('jacks.npz')

## Get starting and end pose and translation
start_pose = full_seq['poses'][44].reshape(55,3) ## The frame 44 was determined by interactive frames in 'amass_vis.py' script
end_pose = full_seq['poses'][104].reshape(55,3)

start_trans = full_seq['trans'][44]
end_trans = full_seq['trans'][104]

## Total t steps
total_steps = 19.0 ## Total steps will be +1 of this
step_size = 1.0 / total_steps
t_list = np.arange(0,1+step_size,step_size)
output_pose = np.zeros((len(t_list), 165))
output_trans = np.zeros((len(t_list), 3))

### convert pose to quaternion
start_rot = Rotation.from_rotvec(start_pose).as_quat()
end_rot = Rotation.from_rotvec(end_pose).as_quat()

### Linearly interpolate the poses
idx = 0
for t in t_list:
    curr_rot = t*end_rot + (1-t)*start_rot
    row_norms = np.linalg.norm(curr_rot, axis=1, keepdims=True)
    curr_rot = curr_rot / row_norms

    curr_trans = t*end_trans + (1-t)*start_trans
    curr_pose = Rotation.from_quat(curr_rot).as_rotvec().reshape(-1)
    
    output_pose[idx] =  curr_pose
    output_trans[idx] = curr_trans
    idx +=1

output_dict = {'trans': output_trans, 'poses': output_pose, 'betas': full_seq['betas'],
               'gender': full_seq['gender']}

np.savez('linear_interpolate.npz', **output_dict)









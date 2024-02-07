import numpy as np
from scipy.spatial.transform import Rotation

start_npy = np.load("npy_files/anna_inter1_smpl_00.npy", allow_pickle=True).item()

start_trans = start_npy['transl']
start_bodypose = start_npy['body_pose'].reshape(21,3)
start_root = start_npy['global_orient'].reshape(1,3)
start_pose = np.zeros((22,3))
start_pose[0,:] =start_root
start_pose[1:,:] = start_bodypose

end_npy = np.load("npy_files/anna_inter2_smpl_00.npy", allow_pickle=True).item()

end_trans = end_npy['transl']
end_bodypose = end_npy['body_pose'].reshape(21,3)
end_root = end_npy['global_orient'].reshape(1,3)
end_pose = np.zeros((22,3))
end_pose[0,:] =end_root
end_pose[1:,:] =end_bodypose

## Total t steps
total_steps = 99.0 ## Total steps will be +1 of this
step_size = 1.0 / total_steps
t_list = np.arange(0,1+step_size,step_size)
output_pose = np.zeros((len(t_list), 66))
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

output_dict = {'trans': output_trans, 'poses': output_pose, 'betas': start_npy['betas'].reshape(-1)[:16],
               'gender': "neutral"}

np.savez('anna_linear_interpolation.npz', **output_dict)
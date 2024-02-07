import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

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
output_pose = np.zeros((len(t_list), 22,3))
output_trans = np.zeros((len(t_list), 3))

for i in range(start_pose.shape[0]):
    start_joint_i = start_pose[i]
    end_joint_i = end_pose[i]
    key_rot = Rotation.from_rotvec([start_joint_i, end_joint_i])
    key_times = [0,1]
    slerp = Slerp(key_times, key_rot)
    interpolate_joint_i = slerp(t_list).as_rotvec()
    output_pose[:, i, :] = interpolate_joint_i

output_pose = output_pose.reshape((len(t_list), -1))

### Linearly interpolate translation
output_trans = np.zeros((len(t_list), 3))
idx = 0
for t in t_list:
    curr_trans = t*end_trans + (1-t)*start_trans
    output_trans[idx] = curr_trans
    idx +=1

output_dict = {'trans': output_trans, 'poses': output_pose, 'betas': start_npy['betas'].reshape(-1)[:16],
               'gender': "neutral"}

np.savez('anna_slerp_interpolate.npz', **output_dict)
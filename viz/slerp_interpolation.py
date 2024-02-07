import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
full_seq = np.load('jacks.npz')

## Get starting and end pose and translation
start_pose = full_seq['poses'][44].reshape(55,3) ## The frame 44 was determined by interactive frames in 'amass_vis.py' script
end_pose = full_seq['poses'][104].reshape(55,3)

start_trans = full_seq['trans'][44]
end_trans = full_seq['trans'][104]

## Total t steps
total_steps = 69.0 ## Total steps will be +1 of this
step_size = 1.0 / total_steps
t_list = np.arange(0,1+step_size,step_size)

### We have to interpolate each joint separately and put it back
output_pose = np.zeros((len(t_list), 55, 3))

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

output_dict = {'trans': output_trans, 'poses': output_pose, 'betas': full_seq['betas'],
               'gender': full_seq['gender']}

np.savez('slerp_interpolate.npz', **output_dict)
 
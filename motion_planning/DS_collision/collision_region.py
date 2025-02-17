import numpy as np
import time
import torch

from motion_planning.NN_model.nn_model_eval import NN_hand_obj

g = [1]

nn = NN_hand_obj(g=g, path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=True)

lb = nn.hand_bound[0, (g[0] - 1) * 4 + 1:g[0] * 4]
ub = nn.hand_bound[1, (g[0] - 1) * 4 + 1:g[0] * 4]

nums = 4

x_ = np.linspace(lb[0], ub[0], nums)
y_ = np.linspace(lb[1], ub[1], nums)
z_ = np.linspace(lb[2], ub[2], nums)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
assert np.all(x[:, 0, 0] == x_)
assert np.all(y[0, :, 0] == y_)
assert np.all(z[0, 0, :] == z_)
x = x.flatten().reshape(-1, 1)
y = y.flatten().reshape(-1, 1)
z = z.flatten().reshape(-1, 1)
q = np.concatenate([np.zeros([nums ** 3, 1]), x, y, z], axis=1)
q_gpu = torch.Tensor(q).to('cuda:0')
print('Generate samples done!')

num_obj = 52
x0 = np.array([[0.06, 0.044, 0.17]])
x_obj = np.repeat(x0, num_obj, axis=0)
x_obj[:, 1] = np.linspace(0.04, 0.06, num_obj)
x_obj_gpu = torch.Tensor(x_obj).to('cuda:0')

t0 = time.time()
output = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=False, only_min_dis=True)
t1 = time.time() - t0
print(q_gpu.shape[0] * x_obj_gpu.shape[0], t1)  # 5000 points, 0.6s

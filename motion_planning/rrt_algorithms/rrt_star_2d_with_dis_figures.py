# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import time
import torch

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

## import NN-based obstacles
from motion_planning.NN_model.nn_model_eval import NN_hand_obj

g = [2]  # index finger

use_cuda = False
nn = NN_hand_obj(g=g, path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=use_cuda)

# the second joints and thirdone for index
lb = nn.hand_bound[0, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]
ub = nn.hand_bound[1, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]

X_dimensions = np.array([tuple([lb[0], ub[0]]), tuple([lb[1], ub[1]])])  # dimensions of Search Space
print('joint limit', X_dimensions)
# obstacles
num_obj = 1
# x0 = np.array([[0.1, 0.044, 0.17]])
# x_obj = np.repeat(x0, num_obj, axis=0)
# x_obj[:, 1] = np.linspace(0.04, 0.06, num_obj)
# x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01]])
# x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)


def get_dis(q, gradient=False, x_obj=None, sample=None):
    # global x_obj_gpu
    if isinstance(q, tuple):
        q = np.array(q)

    # if x_obj is not None:
    x_obj_gpu = x_obj

    if q.shape == (2,):
        q = np.array([0, q[0], q[1], 0])
        q_gpu = torch.Tensor(q.reshape(1,-1)).to('cuda:0') if use_cuda else torch.Tensor(q.reshape(1,-1))
        if gradient:
            output, grad = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=False, only_min_dis=True,
                                   gradient=gradient)  # single q, multiple x
            return output[g[0]].flatten(), grad[g[0]][1:3]
        else:
            # q
            output = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=False, only_min_dis=True, gradient=gradient)
            return output[0]
    elif q.shape[1] == 2:  # nx2
        n = q.shape[0]
        q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
        q = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)
        if gradient:
            output, grad = nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
                                            gradient=gradient)  # multiple q, single x
            return output, grad[:, 1:3]
        else:

            output = nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
                                      gradient=False)  # multiple q,  multiple x
            if sample is None:
                return np.min(output)
            else:
                pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
                         range(int(n / sample))]
                return pairs
    else:
        raise ValueError('q has a wrong shape', q.shape)


# tmp_test = np.array([[0, 0], [1, 1]])
# print(get_dis(tmp_test))
#
#
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0, 0)  # starting location
# x_init = (0., 0)  # starting location
x_goal = (1, 0.25)  # goal location

# q_now = np.array([0, 0,0,0])
q_goal = np.array([0, 1, 0.25, 0])

Q = np.array([(0.5, 4)])  # length of tree edges  (step, )
r = 0.001  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 20  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# DS_step = None
# create Search Space
X = SearchSpace(X_dimensions, get_dis)

safety_margin = 0.001  # meter

from motion_planning.DS_collision.DS import linear_system, Modulation

ds_0 = linear_system(np.array((0,) + x_goal + (0,)))
modify_DS = Modulation(4)
dis_scale = 100
rho = 1

nums = 300
x_ = np.linspace(lb[0], ub[0], nums)
y_ = np.linspace(lb[1], ub[1], nums)
x_grid, y_grid = np.meshgrid(x_, y_, )
x1 = x_grid.flatten().reshape(-1, 1)
y1 = y_grid.flatten().reshape(-1, 1)
q_ = np.concatenate([np.zeros([nums ** 2, 1]), x1, y1, np.zeros([nums ** 2, 1])], axis=1)
q_gpu = torch.Tensor(q_).to('cuda:0') if use_cuda else torch.Tensor(q_)
level = np.linspace(-1, 1, 11)


def get_full_map_dis(x_obj_):
    # input obstacles
    if not isinstance(x_obj_, torch.Tensor):
        x_obj_ = torch.Tensor(x_obj_).to('cuda:0') if use_cuda else torch.Tensor(x_obj_)
    dis = nn.eval_multiple(q_gpu, x_obj_, real_dis=False, only_min_dis=True, gradient=False)
    return dis.reshape(nums, nums)


def get_DS_vel(q, x_obj_gpu):
    if isinstance(q, tuple):
        q = np.array(q)
    if q.shape == (2,):
        q = np.array([0, q[0], q[1], 0])
        q_gpu = torch.Tensor(q.reshape(1, -1)).to('cuda:0') if use_cuda else torch.Tensor(q.reshape(1, -1))
        output, grad = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=True, only_min_dis=True, gradient=True)
        dis = output[0]
        grad = grad[0, 0:4]

        dq = ds_0.eval(q)
        gamma = (dis - safety_margin) * dis_scale + 1
        M = modify_DS.get_M(grad, gamma, dq=dq, rho=rho)
        dq = M @ dq
        return dq[1:3]
    elif q.shape[1] == 2:
        n = q.shape[0]
        q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
        q_tensor = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)

        output, grad_all = nn.eval_multiple(q_tensor, x_obj_gpu, real_dis=True, only_min_dis=True,
                                            gradient=True)  # multiple q, single x
        dq_all = []
        for j in range(n):
            dq = ds_0.eval(q[j, :])
            gamma = (output[j] - safety_margin) * dis_scale + 1
            M = modify_DS.get_M(grad_all[j, :], gamma, dq=dq, rho=rho)
            dq = M @ dq
            dq_all.append(dq[1:3])
        return np.vstack(dq_all)


# create rrt_search
t_record = []
DS_step = 1000  # iteration number of DS solution check.
# DS_step = None
path_cost = []
metric = {'success':0, 'time_cost':0, 'path_len':0 }
success = 0
N = 100
for i in range(N):
    t0 = time.time()
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count, DS_step=DS_step, vel=get_DS_vel,
                  get_dis=get_dis, figures=[x_grid, y_grid, level], get_full_dis=get_full_map_dis)
    path = rrt.rrt_star(variable_step=True, figure=False)
    t1 = time.time() - t0
    t_record.append(t1)
    cost = 0
    if path is not None:
        if np.linalg.norm(np.array(path[-1]) - np.array(x_goal)) < 1e-5:
            success += 1
            print('success, trial=', i)
        for j in range(len(path) - 1):
            cost += np.linalg.norm(np.array(path[j]) - np.array(path[j+1]))
        path_cost.append(cost)
    else:
        print('no feasible path', i)

t_record = np.array(t_record)
print('success rate', success / N)
print('time cost', np.mean(t_record), np.std(t_record))
print('path cost', np.mean(path_cost), np.std(path_cost))

# plot
# plot = Plot("rrt_star_2d")
# plot.plot_tree(X, rrt.trees)
# if path is not None:
#     plot.plot_path(X, path)
# # plot.plot_obstacles(X, Obstacles)
# plot.plot_start(X, x_init)
# plot.plot_goal(X, x_goal)
# plot.draw(auto_open=True)

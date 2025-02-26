# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import time
import torch

from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from motion_planning.PRM.PRM_tools import collision_check

from src.utilities.plotting import Plot

## import NN-based obstacles
from motion_planning.NN_model.nn_model_eval import NN_hand_obj

g = [0, 1, 2, 3, 4]

use_cuda = False
nn = NN_hand_obj(g=g, path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=use_cuda)
x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])
# x_obj += np.array([1.8e-2 * np.sin(step / 20), 0, 0])
x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)

hand = collision_check(x_obj, use_cuda=use_cuda)

# the second joints and third one for index

# lb = hand.nn.hand_bound[0, :]
# ub = hand.nn.hand_bound[1, :]

X_dimensions = hand.nn.hand_bound.T  # dimensions of Search Space


# print('joint limit', X_dimensions)
# obstacles
# num_obj = 1
# x0 = np.array([[0.1, 0.044, 0.17]])
# x_obj = np.repeat(x0, num_obj, axis=0)
# x_obj[:, 1] = np.linspace(0.04, 0.06, num_obj)
# x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01]])
# x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)


def get_dis(q, gradient=False, x_obj=None, sample=None, safety_dis=0):
    # global x_obj_gpu
    if isinstance(q, tuple):
        q = np.array(q).reshape(-1, 16)
    else:
        q = q.reshape(-1, 16)

    q = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)
    # if x_obj is not None:
    # x_obj_gpu = x_obj

    if gradient:
        dis, grad = hand.get_dis(q, x_obj=x_obj, gradient=True, real_dis=True, dx=True)
        return dis, grad
    else:

        if sample is None:

            output1 = hand.get_dis(q=q, sample=sample, safety_dis=safety_dis)
            output2 = hand.SCA_eval(q=q, sample=sample, safety_dis=safety_dis)
            tmp = [min(output1), min(output2)]
            return min(tmp)

        else:
            pairs = hand.collision_hand_obj_SCA(q=q, sample=sample,
                                                safety_dis=safety_dis)  # do the hand-obj collision and SCA
            return pairs

    #
    # if q.shape == (2,):
    #     q = np.array([0, q[0], q[1], 0])
    #     q_gpu = torch.Tensor(q.reshape(1,-1)).to('cuda:0') if use_cuda else torch.Tensor(q.reshape(1,-1))
    #     if gradient:
    #         output, grad = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=False, only_min_dis=True,
    #                                gradient=gradient)  # single q, multiple x
    #         return output[g[0]].flatten(), grad[g[0]][1:3]
    #     else:
    #         # q
    #         output = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=False, only_min_dis=True, gradient=gradient)
    #         return output[0]
    # elif q.shape[1] == 2:  # nx2
    #     n = q.shape[0]
    #     q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
    #     q = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)
    #     if gradient:
    #         output, grad = nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
    #                                         gradient=gradient)  # multiple q, single x
    #         return output, grad[:, 1:3]
    #     else:
    #
    #         output = nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
    #                                   gradient=False)  # multiple q,  multiple x
    #         if sample is None:
    #             return np.min(output)
    #         else:
    #             pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
    #                      range(int(n / sample))]
    #             return pairs
    # else:
    #     raise ValueError('q has a wrong shape', q.shape)


# tmp_test = np.array([[0, 0], [1, 1]])
# print(get_dis(tmp_test))
#
#
q0 = np.zeros(16)
q0[12] = 0.5
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = tuple(q0)  # starting location
# x_init = (0., 0)  # starting location
# x_goal = (1, 0.25)  # goal location
dim = 8
if dim == 8:
    x_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
              0, 0, 0, 0, 0, 0, 0, 0,
              1.1357499, 0.9659528, 1.5200892, 0.6767379)  # goal location
elif dim == 4:
    x_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
              0, 0, 0, 0, 0, 0, 0, 0,
              0.5, 0, 0, 0)
elif dim == 2:
    x_goal = (0, 1.3903904, 1.3273159, 0,
              0, 0, 0, 0, 0, 0, 0, 0,
              0.5, 0, 0, 0)

tmp = np.linspace(np.array(x_init), np.array(x_goal), num=1000)
pairs = get_dis(tmp, safety_dis=0.001, sample=1000)
# q_now = np.array([0, 0,0,0])
q_goal = x_goal

Q = np.array([(0.5, 4)])  # length of tree edges  (step, )
r = 0.001  # length of smallest edge to check for intersection with obstacles
# max_samples = 1024  # max number of samples to take before timing out
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 20  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# DS_step = None
# create Search Space
X = SearchSpace(X_dimensions, get_dis)

safety_margin = 0.001  # meter

from motion_planning.DS_collision.DS import linear_system, Modulation

ds_0 = linear_system(np.array(x_goal))
modify_DS = Modulation(16)
dis_scale = 100
rho = 1

nums = 300


# x_ = np.linspace(lb[0], ub[0], nums)
# y_ = np.linspace(lb[1], ub[1], nums)
# x_grid, y_grid = np.meshgrid(x_, y_, )
# x1 = x_grid.flatten().reshape(-1, 1)
# y1 = y_grid.flatten().reshape(-1, 1)
# q_ = np.concatenate([np.zeros([nums ** 2, 1]), x1, y1, np.zeros([nums ** 2, 1])], axis=1)
# q_gpu = torch.Tensor(q_).to('cuda:0') if use_cuda else torch.Tensor(q_)
# level = np.linspace(-1, 1, 11)


# def get_full_map_dis(x_obj_):
#     # input obstacles
#     if not isinstance(x_obj_, torch.Tensor):
#         x_obj_ = torch.Tensor(x_obj_).to('cuda:0') if use_cuda else torch.Tensor(x_obj_)
#     dis = nn.eval_multiple(q_gpu, x_obj_, real_dis=False, only_min_dis=True, gradient=False)
#     return dis.reshape(nums, nums)


def get_DS_vel(q, x_obj_gpu):
    if isinstance(q, tuple):
        q = np.array(q)
    if q.shape == (16,):
        # q = np.array([0, q[0], q[1], 0])
        q_gpu = torch.Tensor(q.reshape(1, -1)).to('cuda:0') if use_cuda else torch.Tensor(q.reshape(1, -1))
        # output, grad = nn.eval_multiple(q_gpu, x_obj_gpu, real_dis=True, only_min_dis=True, gradient=True)
        # dis = output[0]
        # grad = grad[0, 0:16]
        dis, grad = hand.get_dis(q_gpu, x_obj=x_obj_gpu, gradient=True, real_dis=True, dx=True)

        dq = ds_0.eval(q)
        gamma = (dis - safety_margin) * dis_scale + 1
        M = modify_DS.get_M(grad, gamma, dq=dq, rho=rho)
        dq = M @ dq
        return dq
    elif q.shape[1] == 16:
        n = q.shape[0]
        # q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
        q_tensor = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)

        # output, grad_all = nn.eval_multiple(q_tensor, x_obj_gpu, real_dis=False, only_min_dis=True,
        #                                     gradient=True)  # multiple q, single x
        output, grad_all = hand.get_dis(q_tensor, x_obj=x_obj_gpu, gradient=True, real_dis=True, dx=True)

        dq_all = []
        for j in range(n):
            dq = ds_0.eval(q[j, :])
            gamma = (output[j] - safety_margin) * dis_scale + 1
            M = modify_DS.get_M(grad_all[j, :], gamma, dq=dq, rho=rho)
            dq = M @ dq
            dq_all.append(dq)
        return np.vstack(dq_all)


# create rrt_search
t_record = []
# DS_step = 50  # iteration number of DS solution check.
DS_step = None
path_cost = []
metric = {'success': 0, 'time_cost': 0, 'path_len': 0}
success = 0
N = 100
check = True
for i in range(N):
    t0 = time.time()
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count, DS_step=DS_step, vel=get_DS_vel,
                  get_dis=get_dis, figures=None, get_full_dis=None, dim=dim)
    path = rrt.rrt_star(variable_step=True, figure=False)
    t1 = time.time() - t0
    t_record.append(t1)
    cost = 0
    if path is not None:
        if np.linalg.norm(np.array(path[-1]) - np.array(x_goal)) < 1e-5:
            success += 1
            print('success, trial=', i)
        else:
            print('no feasible path', i)
        path_check = []
        for j in range(len(path) - 1):
            cost += np.linalg.norm(np.array(path[j]) - np.array(path[j + 1]))
            if check:
                tmp = np.linspace(np.array(path[j]), np.array(path[j + 1]), num=300)
                path_check.append(tmp)
        path_cost.append(cost)

        if check:
            path_check = np.vstack(path_check)
            pairs = get_dis(path_check)
            assert pairs
        # check if path is collision free


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

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
x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01]])
x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)


def get_dis(q, gradient=False):
    if isinstance(q, tuple):
        q = np.array(q)

    if q.shape == (2,):
        q = np.array([0, q[0], q[1], 0])
        if gradient:
            output, grad = nn.eval(q, x_obj_gpu, real_dis=False, only_min_dis=True,
                                   gradient=gradient)  # single q, multiple x
            return output[g[0]].flatten(), grad[g[0]][1:3]
        else:
            output = nn.eval(q, x_obj_gpu, real_dis=False, only_min_dis=True, gradient=gradient)
            return output[g[0]]
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
            return np.min(output)
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
r = 0.01  # length of smallest edge to check for intersection with obstacles
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


def get_DS_vel(q):
    if isinstance(q, tuple):
        q = np.array(q)
    if q.shape == (2,):
        q = np.array([0, q[0], q[1], 0])
        output, grad = nn.eval(q, x_obj_gpu, real_dis=True, only_min_dis=True, gradient=True)
        dis = output[g[0]]
        grad = grad[g[0]][0:4]

        dq = ds_0.eval(q)
        gamma = (dis - safety_margin) * dis_scale + 1
        M = modify_DS.get_M(grad, gamma, dq=dq, rho=rho)
        dq = M @ dq
        return dq[1:3]
    elif q.shape[1] == 2:
        n = q.shape[0]
        q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
        q_tensor = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)

        output, grad_all = nn.eval_multiple(q_tensor, x_obj_gpu, real_dis=False, only_min_dis=True,
                                            gradient=True)  # multiple q, single x
        dq_all = []
        for j in range(n):
            dq = ds_0.eval(q[j, :])
            gamma = (output[j] - safety_margin) * dis_scale + 1
            M = modify_DS.get_M(grad_all[j, :], gamma, dq=dq, rho=rho)
            dq = M @ dq
            dq_all.append(dq[1:3])
        return np.vstack(dq_all)


# tmp = (0.7468486833290686, 1.3140411657942734)
# print(get_dis(tmp))
# print(get_DS_vel(tmp))

# t0 = time.time()
# dt = 0.2
# step = 20
# eps = 0.01  # 0.57295779 degree
# x_next = tmp
# b_reached = False
# while step > 0 and not b_reached:
#     dq = get_DS_vel(x_next)
#     x_next = x_next + dq * dt
#     b_reached = np.linalg.norm(x_next - np.array(x_goal)) < eps
#     if b_reached:
#         break
#     step = step - 1
# if step > 0 and b_reached:
#     print('reach the goal', np.linalg.norm(x_next - np.array(x_goal)))
#     t1 = time.time() - t0
#     print('time cost for the DS integral', t1, 10 - step)
# else:
#     print('do not reach the goal', np.linalg.norm(x_next - np.array(x_goal)))
#     print('vel', dq)


# create rrt_search
t_record = []
DS_step = None
for i in range(100):
    t0 = time.time()
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count, DS_step=DS_step, vel=get_DS_vel,
                  get_dis=get_dis)
    path = rrt.rrt_star(variable_step=False)
    t1 = time.time() - t0
    t_record.append(t1)
    print(i)
t_record = np.array(t_record)
print('time cost', np.mean(t_record), np.std(t_record))

# plot
plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
# plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)

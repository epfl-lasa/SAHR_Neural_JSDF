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
#
#
q0 = np.zeros(16)
q0[12] = 0.5
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = tuple(q0)  # starting location
# x_init = (0., 0)  # starting location
x_goal = (1, 0.25)  # goal location
x_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
          -0.06984199999999996, 1.3148996000000002, 1.27591, 0.43,
          0.0, 1.3293476000000002, 1.2510544000000001, 0.55,
          1.1357499, 0.9659528, 1.5200892, 0.6767379)  # goal location
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
                  get_dis=get_dis, figures=None, get_full_dis=None)
    path = rrt.rrt_star(variable_step=True, figure=False)
    t1 = time.time() - t0
    t_record.append(t1)
    cost = 0
    if path is not None:
        if np.linalg.norm(np.array(path[-1]) - np.array(x_goal)) < 1e-5:
            success += 1
            print('success, trial=', i)
        path_check = []
        for j in range(len(path) - 1):
            cost += np.linalg.norm(np.array(path[j]) - np.array(path[j + 1]))
            if check:
                tmp = np.linspace(np.array(path[j]), np.array(path[j + 1]), num=100)
                path_check.append(tmp)
                assert np.all(path[j]>= hand.nn.hand_bound[0,:])
                assert np.all(path[j]<= hand.nn.hand_bound[1,:])
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

### plot
# plot = Plot("rrt_star_2d")
# plot.plot_tree(X, rrt.trees)
# if path is not None:
#     plot.plot_path(X, path)
# # plot.plot_obstacles(X, Obstacles)
# plot.plot_start(X, x_init)
# plot.plot_goal(X, x_goal)
# plot.draw(auto_open=True)

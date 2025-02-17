# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import time
import torch

from src.rrt.rrt_star import RRTStar
from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

## import NN-based obstacles
from motion_planning.NN_model.nn_model_eval import NN_hand_obj

g = [1]  # index finger

use_cuda = False
nn = NN_hand_obj(g=g, path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=use_cuda)

# the second joints and thirdone for index
lb = nn.hand_bound[0, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]
ub = nn.hand_bound[1, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]

X_dimensions = np.array([tuple([lb[0], ub[0]]), tuple([lb[1], ub[1]])])  # dimensions of Search Space

# obstacles
num_obj = 5
x0 = np.array([[0.1, 0.044, 0.17]])
x_obj = np.repeat(x0, num_obj, axis=0)
x_obj[:, 1] = np.linspace(0.04, 0.06, num_obj)
x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)


def get_dis(q):
    if isinstance(q, tuple):
        q = np.array(q)

    if q.shape == (2,):
        q = np.array([0, q[0], q[1], 0])
        output = nn.eval(q, x_obj_gpu, real_dis=False, only_min_dis=True, gradient=False)
        return output[g[0]]
    elif q.shape[1] == 2:  # nx2
        n = q.shape[0]
        q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
        q = torch.Tensor(q).to('cuda:0') if use_cuda else torch.Tensor(q)
        output = nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True, gradient=False)
        return np.min(output)
    else:
        raise ValueError('q has a wrong shape', q.shape)



tmp_test = np.array([[0, 0], [1, 1]])
print(get_dis(tmp_test))
#
#
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0.6, 0.5)  # starting location
# x_init = (0., 0)  # starting location
x_goal = (1.5, 1)  # goal location

Q = np.array([(0.1, 4)])  # length of tree edges
r = 0.01  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 0  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, get_dis)

#
# create rrt_search
rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()

# plot
plot = Plot("rrt_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
# plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)

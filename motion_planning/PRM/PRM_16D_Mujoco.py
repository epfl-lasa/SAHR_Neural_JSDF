import time

import mujoco
from mujoco import viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import controller_utils
import tools.rotations as rot
import torch
import time
from D_star_lite import DStar

from KNN_test import FaissKNeighbors

from PRM_tools import collision_check, graph

model = mujoco.MjModel.from_xml_path("../../description/iiwa7_lasa_allegro_new_prm.xml") # this is allegro right hand

data = mujoco.MjData(model)
view = viewer.launch_passive(model, data)

r = controller_utils.Robot(model, data, view, obj_names=['box1','box2','box3','box4'])

x0 = np.copy(r.x)  # the initial iiwa pose

# initialize the env

g = [0, 1, 2, 3, 4]  # palm, index/middle/ring/thumb finger
use_cuda = True  # if the NVIDIA driver and cuda are well installed, we can use cuda to speed up the collision detection
                 # by NN, otherwise, use CPU by setting False
                 # if GPU out of memory, set a small $n$ and $k_0$

x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])

hand = collision_check(x_obj, use_cuda=use_cuda, right_hand=True)

# generate samples
dim = 16
n = 2000

# build graph

graph_16D = graph(dim=dim)

s_start = tuple(r.qh)  # starting location

s_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
          -0.06984199999999996, 1.3148996000000002, 1.27591, 0.43,
          0.0, 1.3293476000000002, 1.2510544000000001, 0.55,
          1.1357499,   0.9659528, 1.5200892, 0.6767379)  # goal location

# k-NN nearest
t0 = time.time()
k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 7
edges = {}  # to be checked if collision-free
edge_sample_num = 20

# step = 0
step_max = 300

# D star

pairs_last = []
step = 0
s_v = 0.02
qh_ref = np.copy(np.array(s_goal))
optimize = False
moving = False
first = True
safety_dis = 0.0001
while 1:
    t0 = time.time()
    # x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])  # old
    # x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])
    if first:
        x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187], [0.11, 0.03, 0.1]]) # four obstacles
        # if we have a complex obstacle, a representation by point-cloud or spheres are required

        # x_obj = np.array([[800.10000000e-02, 0, 100.87000000e-01]])  # test

        # x_obj = np.array([[0.08, -0.02, 0.187]])
        # x_obj = np.array([[0.08, 0.02, 0.187]])
        # x_obj[:, 1] += 1.8e-2 * np.sin(step / 400)
        # x_obj[:, 1] += 5e-2 * np.sin(step / 200)
        x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
        hand.x_obj_gpu = x_obj_gpu

        for i in range(len(x_obj)):
            r.d.mocap_pos[i, :] = r.p[:3, :3] @ x_obj[i, :] + r.x[:3]  # move the object/obstacle

        # samples
        samples = np.random.uniform(hand.nn.hand_bound[0, :], hand.nn.hand_bound[1, :], size=(n, dim))  # (n, dim)
        samples = np.concatenate(
            [np.array(s_start).reshape(1, -1), np.array(s_goal).reshape(1, -1), samples])  # (n+2, dim)
        knn = FaissKNeighbors(k=k0 + 1)  # knn
        knn.fit(samples)
        samples_near = knn.predict(samples)[:, 1:, :]  # remove the first one, which is itself
        s1 = list(map(tuple, samples))
        s2 = [list(map(tuple, samples_near[i, :, :])) for i in range(samples_near.shape[0])]

        graph_16D.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
        d_star = DStar(s_start, s_goal, graph_16D, "euclidean")
        start_p = np.repeat(samples, repeats=k0, axis=0)
        edge_samples_ = np.linspace(start_p, np.vstack(samples_near),
                                    edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
        edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

        # pairs = hand.get_dis(edge_samples, gradient=False, sample=edge_sample_num, safety_dis=safety_dis/5, real_dis=True)
        hand.q_gpu = torch.Tensor(edge_samples).to('cuda:0') if use_cuda else torch.Tensor(edge_samples)
        pairs = hand.collision_hand_obj_SCA(q=None, sample=edge_sample_num, safety_dis=safety_dis)  # do the hand-obj collision and SCA
        start_end_p = []
        for i in range(len(s2)):
            for j in range(k0):
                start_end_p.append(s1[i] + s2[i][j])

    t1 = time.time()
    if first or moving:
        pairs = hand.collision_hand_obj_SCA(q=None, sample=edge_sample_num, safety_dis=safety_dis)
        d_star.graph.E = dict(zip(start_end_p, pairs))

    # print('time cost of collision check', time.time() - t1)
                                                    # each edge is representated by a tuple of two vertices
    d_star.graph.E = dict(zip(start_end_p, pairs))  # build the dict for collision state of each edge {v1 + v2 : False/True, ...}

    if moving:
        if len(pairs_last) != 0:
            diff = np.logical_xor(pairs, pairs_last)  # True means the state is changed
            # store the edges whose costs need to update in the PRM graph
            edges2update = [start_end_p[i] for i in range(len(start_end_p)) if diff[i]]
            if len(edges2update):
                print('edge num to be updated', len(edges2update))
                d_star.update_cost(edges2update)
    pairs_last = pairs

    # t0 = time.time()
    if first or moving:
        result = d_star.ComputePath()
    # qh_ref = np.copy(r.qh)

    if result:

        path = d_star.extract_path()
        # print('find path', len(path))
        # direction = np.array(path[1]) - r.qh
        # s_robot = r.qh + s_v * direction / (np.linalg.norm(direction) + 8e-3)
        if optimize and path[-1] == s_goal:
            # print('p')
            s1 = path
            s2 = [path[:i] + path[i + 1:] for i in range(len(path))]
            graph_2D_opt = graph(dim=dim)
            graph_2D_opt.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
            d_star_opt = DStar(s_start, s_goal, graph_2D_opt, "euclidean")
            samples_opt = np.array(s1)
            start_p_opt = np.repeat(samples_opt, repeats=len(s1) - 1, axis=0)
            num_tmp = 200
            edge_samples_ = np.linspace(start_p_opt, np.vstack(s2),
                                        num_tmp, axis=1)  # (n*k, edge_sample_num, dim)
            edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)
            hand.q_gpu = torch.Tensor(edge_samples).to('cuda:0') if use_cuda else torch.Tensor(edge_samples)

            pairs2 = hand.collision_hand_obj_SCA(sample=num_tmp, safety_dis=safety_dis)
            start_end_p = []
            for i in range(len(s1)):
                for j in range(len(s1) - 1):
                    start_end_p.append(s1[i] + s2[i][j])
            d_star_opt.graph.E = dict(zip(start_end_p, pairs2))
            # print('knn + update graph', time.time() - t0)
            # update cost
            result_opt = d_star_opt.ComputePath()

            path_opt = d_star_opt.extract_path(max_nums=1000)
            time_cost = time.time() - t0

            path = path_opt
        qh_ref = np.array(path[1])
        dis = np.linalg.norm(r.qh - qh_ref)

        # check the path
        num_tmp = 200
        edge_samples_path = np.linspace(np.vstack(path[0:-1]), np.vstack(path[1:]),
                                    num_tmp, axis=1)  # (n*k, edge_sample_num, dim)
        edge_samples_path = np.vstack(edge_samples_path)
        hand.q_gpu = torch.Tensor(edge_samples_path).to('cuda:0') if use_cuda else torch.Tensor(edge_samples_path)
        pairs2 = hand.collision_hand_obj_SCA(sample=num_tmp, safety_dis=safety_dis)

        # print(dis)
        if dis < 0.3:
            d_star.s_start = path[1]
            d_star.km += d_star.h(path[0], path[1])
            print('robot reached the next node. Step', step)
            path.pop(0)
            if path[1] == s_goal:
                print('Reach the goal')
                break
        # print(x0 - r.x)
        r.iiwa_hand_go(np.copy(x0), qh_ref, kh_scale=0.2 * np.ones(4))
        step += 1
        # print(time.time() - t0)
        time.sleep(0.03)
        # print('Find a feasible path')
        first = False
    else:
        path = []
        # print('No feasible path')
        first = True
    #
    # # # send command to iiwa and hand
    # r.moveto_attractor(x0, qh_ref, couple_hand=False, scaling=2)
    #
    # t1 = time.time() - t0
    # print(t1)
    # r.iiwa_hand_go(x0, qh_ref, kh_scale= [5,5,5,5])


while 1:
    qh_ref = np.array(path[1])
    r.iiwa_hand_go(np.copy(x0), qh_ref, kh_scale= [0.2,0.2,0.2,0.2])
    time.sleep(0.002)

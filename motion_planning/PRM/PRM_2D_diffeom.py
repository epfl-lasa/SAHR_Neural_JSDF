import numpy as np
import time
import torch
from rtree import index
import time, os

from KNN_test import FaissKNeighbors
from PRM_tools import collision_check, graph
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

from matplotlib import rc
from matplotlib.font_manager import FontProperties


import motion_planning.diffeomorphism.position_mapping_N_D as pm

import motion_planning.DS_collision.DS as DS
# initialize the env


g = [2]  # middle finger
use_cuda = True

x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])

collision = collision_check(x_obj, g=g, use_cuda=use_cuda, )
# the second joints and third one for index

lb = collision.nn.hand_bound[0, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]
ub = collision.nn.hand_bound[1, (g[0] - 1) * 4 + 1:g[0] * 4 - 1]

# generate samples
dim = 2
n = 500
samples = np.random.uniform(lb, ub, size=(n, dim))

samples_col_bool = collision.obstacle_free(samples)

# samples_free = samples[samples_col_bool, :]  # use only collision-free samples
samples_free = samples  # use all samples
print('ratio of collision-free samples', samples_free.shape[0] / n)

# build graph

graph_2D = graph(dim=dim)
sample_free_tuple = list(map(tuple, samples_free))
s_start = (0, 0)  # starting location
s_goal = (1, 0.25)  # goal location
sample_free_tuple.append(s_start)
sample_free_tuple.append(s_goal)
samples = np.concatenate([np.array(s_start).reshape(1, -1), np.array(s_goal).reshape(1, -1), samples])

# # k-NN nearest
# t0 = time.time()
k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 5
# edges = {}  # to be checked if collision-free
edge_sample_num = 10

from D_star_lite import DStar

# def clip_samples(data):
#     data[:]

# D star

pairs_last = []
s_robot = np.array(s_start)
s_v = 0.04
step = 0
step_max = 100
max_nums = 1000
safety_dis = 5e-3
dt = 0.02

metrics = {'success': [], 'time_cost': [], 'path_len': []}

optimize = True
dynamic = False
first = True
rec_band = []

DS_M = DS.Modulation(len(s_start))
while 1:
    # samples = np.random.uniform(lb, ub, size=(n, dim))
    # samples = np.concatenate([np.array(s_start).reshape(1, -1), np.array(s_goal).reshape(1, -1), samples])

    if step >= step_max:
        break
    print('step', step)
    t_all = time.time()
    # x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
    # x_obj = np.array([[10.10000000e-02, 0, 1.87000000e-01]])

    # x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01],])   # this one is consistent with the paper. narrow passage
    x_obj = np.array([[9.90000000e-02, 0, 1.97000000e-01], ])  # this one is a test point, bigger collision-free space
    # x_obj += np.array([1.8e-2 * np.sin(step / 20), 0, 0])   # move the obstacle
    # x_obj += np.array([1.8e-2 * np.sin(step / 30), 0, 0])   # move the obstacle
    # x_obj = np.array([[8.9000000e-02, 0, 1.97000000e-01],
    #                   [7.0000000e-02, 0, 1.400000e-01]])  # connected two obstacles
    # x_obj = np.array([[8.9000000e-02, 0, 1.97000000e-01],
    #                   [12.0000000e-02, 0, 1.500000e-01]])  # connected two obstacles. narrow path, 2 local minimum

    x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
    collision.x_obj_gpu = x_obj_gpu

    t0 = time.time()
    # pairs = collision.get_dis(edge_samples, gradient=False, sample=edge_sample_num, safety_dis=safety_dis)
    # print('edge collision check time:', time.time() - t0)

    # d_star.graph.E = dict(zip(start_end_p, pairs))  # {v1 + v2 : False/True, ...}
    #
    # if len(pairs_last) != 0:
    #     diff = np.logical_xor(pairs, pairs_last)  # True means the state is changed
    #     # store the edges that need to update
    #     edges2update = [start_end_p[i] for i in range(len(start_end_p)) if diff[i]]
    #     # print('edge num to be updated', len(edges2update))
    #     d_star.update_cost(edges2update)
    # pairs_last = pairs
    if first:
        if dynamic:
            t0 = time.time()
            static_node_num = -250
            for i in range(5):  # update samples to make them closer to the safety_dis
                dis, grad = collision.get_dis(samples, gradient=True, real_dis=True, dx=True)
                dx = grad[:, 4:]
                # x_change = np.array([1.8e-2 * np.sin(step / 20), 0, 0]) - np.array([1.8e-2 * np.sin((step-1) / 20), 0, 0])

                # samples[2:static_node_num, :] = samples[2:static_node_num, :] - (dis[2:static_node_num].reshape(-1, 1) - safety_dis * 10 - dx[2:static_node_num] @ x_change.reshape(-1,1)) / grad[2:static_node_num, 1:3] * 0.01 * 3
                samples[2:static_node_num, :] = samples[2:static_node_num, :] - (
                        dis[2:static_node_num].reshape(-1, 1) - safety_dis * 4) / grad[2:static_node_num,
                                                                                  1:3] * 0.01 * 3
                samples[:static_node_num, 0] = np.clip(samples[:static_node_num, 0], lb[0], ub[0])
                samples[:static_node_num, 1] = np.clip(samples[:static_node_num, 1], lb[1], ub[1])
            print('time cost for update samples', time.time() - t0)

        # update graph
        # t0 = time.time()
        t_knn = time.time()
        knn = FaissKNeighbors(k=k0 + 1)  # knn
        knn.fit(samples)
        samples_near = knn.predict(samples)[:, 1:, :]  # remove the first one, which is itself
        t_knn = time.time() - t_knn
        print('KNN time', t_knn)
        s1 = list(map(tuple, samples))
        s2 = [list(map(tuple, samples_near[i, :, :])) for i in range(samples_near.shape[0])]

        graph_2D.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
        path = []
        for i in range(1):
            if len(path) != 0 and np.linalg.norm(np.array(path[-1]) - np.array(s_goal)) < 1e-5:
                break

            t1 = time.time()
            d_star = DStar(s_start, s_goal, graph_2D, "euclidean")
            # print("Initialization:", time.time() - t1)
            start_p = np.repeat(samples, repeats=k0, axis=0)
            edge_samples_ = np.linspace(start_p, np.vstack(samples_near),
                                        edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
            edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

            t2 = time.time()
            pairs = collision.get_dis(edge_samples, gradient=False, sample=edge_sample_num, safety_dis=safety_dis,
                                      real_dis=True)
            # print("Collision check:", time.time() - t2)
            start_end_p = []
            for i in range(len(s2)):
                for j in range(k0):
                    start_end_p.append(s1[i] + s2[i][j])
            d_star.graph.E = dict(zip(start_end_p, pairs))
            # print('knn + update graph', time.time() - t0)
            # update cost
            t3 = time.time()
            result = d_star.ComputePath()
            print("update cost:", time.time() - t3)

            t4 = time.time()
            path = d_star.extract_path(max_nums=max_nums)
            print("extract path cost:", time.time() - t4)
            # print("D start time cost:", time.time() - t1)
        # if result:
        #     path = d_star.extract_path(max_nums=max_nums)
        # else:
        #     path = []
        # print('step', step, len(path), '   find path cost:', time.time() - t0)
        print(time.time() - t0)
        time_cost = time.time() - t0
        if np.linalg.norm(np.array(path[-1]) - np.array(s_goal)) < 1e-5:
            a = 1
            print('Step:', step, '  Success, time all', time_cost)
            metrics['success'].append(1)

            if optimize:
                t_opt = time.time()
                s1 = path
                s2 = [path[:i] + path[i + 1:] for i in range(len(path))]
                graph_2D_opt = graph(dim=dim)
                graph_2D_opt.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
                d_star_opt = DStar(s_start, s_goal, graph_2D_opt, "euclidean")
                samples_opt = np.array(s1)
                start_p_opt = np.repeat(samples_opt, repeats=len(s1) - 1, axis=0)
                edge_sample_num_tmp = 300
                edge_samples_ = np.linspace(start_p_opt, np.vstack(s2),
                                            edge_sample_num_tmp, axis=1)  # (n*k, edge_sample_num, dim)
                edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

                pairs = collision.get_dis(edge_samples, gradient=False, sample=edge_sample_num_tmp,
                                          safety_dis=safety_dis,
                                          real_dis=True)
                start_end_p = []
                for i in range(len(s1)):
                    for j in range(len(s1) - 1):
                        start_end_p.append(s1[i] + s2[i][j])
                d_star_opt.graph.E = dict(zip(start_end_p, pairs))
                # print('knn + update graph', time.time() - t0)
                # update cost
                result_opt = d_star_opt.ComputePath()

                path_opt = d_star_opt.extract_path(max_nums=max_nums)
                print('opt time', time.time() - t_opt)
                time_cost = time.time() - t0

                path = path_opt  # a list of tuple

            metrics['time_cost'].append(time_cost)
            path_len = 0
            for j in range(len(path) - 1):
                path_len += np.linalg.norm(np.array(path[j]) - np.array(path[j + 1]))
            metrics['path_len'].append(path_len)

            first = False
        else:
            metrics['success'].append(0)
            print('No feasible path')

            first = True

        # (1) sample nodes on the path
        if metrics['success'][-1]:
            nodes = []
            sample_interval = 0.05 / 2
            for i in range(len(path) - 1):
                p = np.vstack([np.array(path[i]), np.array(path[i + 1])])
                length_p = np.linalg.norm(p[0, :] - p[1, :])
                num = int(length_p / sample_interval) + 1  # >=1
                node = np.linspace(p[0, :], p[1, :], num)  # (num, 2)
                nodes.append(node)

            nodes = np.vstack(nodes)
            print('Full node number:', nodes.shape[0])
            para = np.array([50, 0.6, 0.95])  # iteration algorithm parameter
            model = pm.iteration(para, nodes, modulation=True,  get_dis=collision.get_dis)

        else:
            nodes = None

    else:
        pass

    if nodes is not None:
        # para = np.array([50, 0.6, 0.95])  # iteration algorithm parameter
        # model = pm.iteration(para, nodes, modulation=True,  get_dis=collision.get_dis)
        fig, ax = model.visualize(bounds=collision.nn.hand_bound[:, (g[0] - 1) * 4 + 1:g[0] * 4 - 1])

        # fig, ax = model.visualize()

    # figures

    figure = True
    if figure and nodes is not None:
        # save_path = 'figures/elastic_band/exp_moving/'
        paper_path = 'figures/diffeom/2_objs/'
        os.makedirs(paper_path, exist_ok=True)

        output = collision.get_full_map_dis()
        con = ax[2].contourf(collision.x_grid, collision.y_grid, output, np.linspace(-1, 1, 11), cmap='PiYG')

        ax[2].contour(con, levels=[0], colors=('k',), linestyles=('-',),
                    linewidths=(2,))  # boundary of the obs in C space
        # fm = FontProperties(weight='bold')
        # fm = FontProperties(fo)
        # ax[2].set_xticklabels(fontproperties=fm)
        if optimize:
            title = 'Isolines and PRM*'
        else:
            title = 'Isolines and DS streamlines'
        # plt.title(label=title)
        cax = plt.axes([0.92, 0.1, 0.02, 0.75])
        plt.colorbar(con, cax=cax)
        plt.title(label='Dis')

        fig.savefig(paper_path + 'PRM_diffeom_DS_' + str(step) + '.png', format='png', bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300)
        plt.close()

    step += 1

print("success rate:", sum(metrics['success']) / len(metrics['success']))
print('time cost:', np.mean(metrics['time_cost']), np.std(metrics['time_cost']))
print('Path len:', np.mean(metrics['path_len']), np.std(metrics['path_len']))
print("elastic band time cost:", np.mean(rec_band), np.std(rec_band))

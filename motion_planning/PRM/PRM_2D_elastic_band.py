import numpy as np
import time
import torch
from rtree import index
import time, os

from KNN_test import FaissKNeighbors
from PRM_tools import collision_check, graph
import matplotlib.pyplot as plt
from matplotlib import rc

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
s_goal = (1.25, 0.25)  # goal location
sample_free_tuple.append(s_start)
sample_free_tuple.append(s_goal)
samples = np.concatenate([np.array(s_start).reshape(1, -1), np.array(s_goal).reshape(1, -1), samples])
#
# [graph_2D.add_vertex(sample) for sample in sample_free_tuple]
#
# # k-NN nearest
# t0 = time.time()
k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 5
# edges = {}  # to be checked if collision-free
edge_sample_num = 10
#
# # collision check for edges
# t0 = time.time()
#
# start_p = []
# end_p = []
# start_end_p = []
# for i in range(len(sample_free_tuple)):
#     q = sample_free_tuple[i]
#     if q in [s_start, s_goal]:
#         k = k0 + 20
#     else:
#         k = k0
#     near_q = graph_2D.nearby(q, k)
#     near_q_list = []
#     for near_j in near_q:
#         if near_j == q:
#             continue
#         start_p.append(q)
#         start_p.append(near_j)
#         end_p.append(near_j)
#         end_p.append(q)
#         near_q_list.append(near_j)
#         start_end_p.append(q + near_j)
#         start_end_p.append(near_j + q)
#         if near_j in edges:
#             if q not in edges[near_j]:
#                 edges[near_j].append(q)
#         else:
#             edges[near_j] = [q]
#     if q in edges:
#         edges[q] += near_q_list
#     else:
#         edges[q] = near_q_list
#
#     # assert len(near_q_list) == k
#     assert q not in edges[q]
#
# edge_samples_ = np.linspace(np.vstack(start_p), np.vstack(end_p),
#                             edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
# edge_samples = np.vstack(edge_samples_)
# print(edge_samples.shape)
# print('K-NN time:', time.time() - t0)

# step = 0
# step_max = 300
# graph_2D.edges = edges  # {v1: [a1,a2,a3], v2; [...],...}

from D_star_lite import DStar

# def clip_samples(data):
#     data[:]

# D star

pairs_last = []
s_robot = np.array(s_start)
s_v = 0.04
step = 0
step_max = 400
max_nums = 1000
safety_dis = 5e-3
dt = 0.02

metrics = {'success': [], 'time_cost': [], 'path_len': []}

optimize = True
dynamic = False
first = True
rec_band = []
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
    x_obj += np.array([1.8e-2 * np.sin(step / 30), 0, 0])   # move the obstacle
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
            print('Success, time all', time_cost)
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
        else:
            nodes = None

    else:

        if nodes is not None:
            for band_update_num in range(20):
                t_band0 = time.time()

                Kr = 1  # 0.2 for exp
                Kc = 1  # 40 for exp function
                ratio = 1.4
                alpha = 0.2  # 0.01 for exp

                kr = 0.01
                kc = 100 * 3
                alpha = 0.05

                beta = 1000
                pass

                # (2) build the external repulsion force by the distance and gradient from NN
                dis, grad = collision.get_dis(nodes, gradient=True, real_dis=True, dx=True)
                dq = grad[:, 1:3]
                if np.all(dis):
                    # F_r = Kr * (ratio * safety_dis - dis.reshape(-1, 1)) * dq
                    F_r = Kr * np.exp((ratio * safety_dis - dis.reshape(-1, 1)) * beta) * dq
                    # F_r[dis > ratio * safety_dis, :] = 0

                    # (3) build internal contraction force
                    # F_c0 = (nodes[1:, :] - nodes[:-1, :]) / np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1).reshape(-1,1)  # (num-1,2)

                    F_c0 = nodes[1:, :] - nodes[:-1, :]  # (num-1,2)
                    F_c1 = np.concatenate([F_c0, np.zeros([1, 2])])  # (num, 2), the last node is applied zero force

                    F_c2 = np.concatenate([np.zeros([1, 2]), - F_c0])  # (num, 2)

                    # enhance the force for the first one and the last one
                    F_c1[-2, :] = F_c1[-2, :] * 40
                    F_c2[1, :] = F_c2[1, :] * 40

                    F_c = Kc * (F_c1 + F_c2)

                    # (4) sum the total force and move all nodes
                    F = F_c + F_r  # the sum of forces

                    # the start and end point keep fixed
                    F[0, :] = 0
                    F[-1, :] = 0

                    nodes += alpha * F

                    # to make the nodes stay within the joint limit
                    nodes[:, 0] = np.clip(nodes[:, 0], lb[0], ub[0])
                    nodes[:, 1] = np.clip(nodes[:, 1], lb[1], ub[1])

                    ## add nodes between two far way nodes
                    nodes_norm = np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1)
                    bool_dis = nodes_norm > sample_interval * 2
                    big_dis_index = [-1] + list(np.where(bool_dis)[0])
                    if sum(bool_dis):
                        nodes_tmp = []
                        for i in range(1, len(big_dis_index)):
                            j = big_dis_index[i]
                            nodes_before = nodes[big_dis_index[i - 1] + 1: j+1, :]
                            intermediate_node = (nodes[j:j + 1, :] + nodes[j + 1:j + 2, :]) / 2
                            nodes_tmp.append(nodes_before)
                            nodes_tmp.append(intermediate_node)
                        nodes_tmp.append(nodes[big_dis_index[-1] + 1:, :])

                        nodes_tmp = np.vstack(nodes_tmp)
                        nodes = nodes_tmp

                    ## remove nodes which are too close to the start/end point
                    # removed_node = False
                    # nodes_norm = np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1)
                    # if nodes_norm[0] < sample_interval/2 * 1.8:
                    #     nodes = np.delete(nodes, 1, axis=0)
                    #     removed_node = True
                    #     print('remove 2nd node')
                    # if nodes_norm[-1] < sample_interval/2 * 1.8:
                    #     nodes = np.delete(nodes, -2, axis=0)
                    #     removed_node = True
                    #     print('remove -2nd node')
                    #
                    # if np.linalg.norm(nodes[0,:] - np.array(s_start))> 1e-8:
                    #     print("start point has moved")

                    ## remove nodes which are too close to each other
                    nodes_norm = np.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1)
                    bool_dis_close = list(np.where(nodes_norm[:-1] < sample_interval/2 * 1.8)[0] +1)
                    nodes = np.delete(nodes, bool_dis_close, axis=0)
                    if nodes_norm[-1] < sample_interval/2 * 1.8:
                        nodes = np.delete(nodes, -2, axis=0)

                else:
                    first = True
                    print('Some nodes are in collision')

                t_band = time.time() - t_band0
                rec_band.append(t_band)
    # figures

    figure = True
    if figure:
        save_path = 'figures/elastic_band/exp_moving/'
        os.makedirs(save_path, exist_ok=True)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = "8"
        rc('text', usetex=True)

        fig_width = 90 / 25.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        output = collision.get_full_map_dis()
        con = plt.contourf(collision.x_grid, collision.y_grid, output, np.linspace(-1, 1, 11), cmap='PiYG')

        plt.contour(con, levels=[0], colors=('k',), linestyles=('-',),
                    linewidths=(2,))  # boundary of the obs in C space
        if optimize:
            title = 'Isolines and PRM*'
        else:
            title = 'Isolines and PRM'
        # plt.title(label=title)
        cax = plt.axes([0.95, 0.1, 0.05, 0.8])
        plt.colorbar(con, cax=cax)
        plt.title(label='Dis')

        ax.set_xlabel('$q_1$' + ' (rad)')
        ax.set_ylabel('$q_2$' + ' (rad)')
        # for k, v in d_star.graph.edges.items():
        #     for vi in v:
        #         if d_star.graph.E[k + vi]:
        #             ax.plot([k[0], vi[0]], [k[1], vi[1]], color='gray')
        # ax.scatter(samples[:, 0], samples[:, 1], c='k', zorder=100, s=10)  # all nodes of PRM
        ax.scatter(d_star.s_start[0], d_star.s_start[1], c='b', zorder=100)
        ax.scatter(s_goal[0], s_goal[1], c='r', zorder=100)
        # ax.scatter(s_robot[0], s_robot[1], c='k', zorder=100)
        # plot feasible path
        ax.set_aspect('equal', adjustable='box')

        # if 1000 > len(path) > 1:
        #     if optimize:
        #         path = path_opt
        #     for i in range(len(path) - 1):
        #         ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='r')

        if nodes is not None:
            ax.plot(nodes[:, 0], nodes[:, 1], color='r')

        fig.savefig(save_path + 'PRM_elastic_band_' + str(step) + '.png', format='png', bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300)
        plt.close()

    step += 1

print("success rate:", sum(metrics['success']) / len(metrics['success']))
print('time cost:', np.mean(metrics['time_cost']), np.std(metrics['time_cost']))
print('Path len:', np.mean(metrics['path_len']), np.std(metrics['path_len']))
print("elastic band time cost:", np.mean(rec_band), np.std(rec_band))

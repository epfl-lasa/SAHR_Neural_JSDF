import numpy as np
import time
import torch
from rtree import index
import time, os

from KNN_test import FaissKNeighbors
from PRM_tools import collision_check, graph
import matplotlib.pyplot as plt

# initialize the env


g = [0,1,2,3,4]  # palm, index/middle/ring/thumb finger
use_cuda = True

x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
hand = collision_check(x_obj, use_cuda=use_cuda)

# generate samples
dim = 16
n = 1000
samples = np.random.uniform(hand.nn.hand_bound[0, :], hand.nn.hand_bound[1, :], size=(n, dim))

# build graph

graph_16D = graph(dim=dim)
q0 = np.zeros(16)
q0[12] = 0.5
s_start = tuple(q0)  # starting location

s_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
          -0.06984199999999996, 1.3148996000000002, 1.27591, 0.43,
          0.0, 1.3293476000000002, 1.2510544000000001, 0.55,
          1.1357499,   0.9659528, 1.5200892, 0.6767379)  # goal location

edge_sample_num = 10


test_traj = np.linspace(s_start, s_goal, edge_sample_num, axis=1).reshape(-1, 16)
x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])

x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
hand.x_obj_gpu = x_obj_gpu

# test_result = hand.get_dis(test_traj, gradient=False, sample=edge_sample_num, safety_dis=0.0, real_dis=True)



k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 5
k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 15


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


metrics = {'success':[], 'time_cost':[], 'path_len':[] }

optimize = False
dynamic = False
while 1:
    samples = np.random.uniform(hand.nn.hand_bound[0, :], hand.nn.hand_bound[1, :], size=(n, dim))  # (n, dim)
    samples = np.concatenate([np.array(s_start).reshape(1, -1), np.array(s_goal).reshape(1, -1), samples])  # (n+2, dim)

    if step >= step_max:
        break
    print('step', step)
    t_all = time.time()
    x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])
    # x_obj += np.array([1.8e-2 * np.sin(step / 20), 0, 0])
    # x_obj = np.array([[9.90000000e-02, 0, 1.97000000e-01],])   # this one is a test point, bigger collision-free space
    x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
    hand.x_obj_gpu = x_obj_gpu

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
    if dynamic:
        t0 = time.time()
        static_node_num = 148
        for i in range(3):  # update samples to make them closer to the safety_dis
            dis, grad = hand.get_dis(samples, gradient=True, real_dis=True, dx=True)
            # dx = grad[:, 4:]
            # x_change = np.array([1.8e-2 * np.sin(step / 20), 0, 0]) - np.array([1.8e-2 * np.sin((step-1) / 20), 0, 0])

            # samples[2:static_node_num, :] = samples[2:static_node_num, :] - (dis[2:static_node_num].reshape(-1, 1) - safety_dis * 2 - dx[2:static_node_num] @ x_change.reshape(-1,1)) / grad[2:static_node_num, 1:3] * 0.01 * 3
            samples[2:static_node_num, :] = samples[2:static_node_num, :] - (dis[2:static_node_num].reshape(-1, 1) - safety_dis * 2) / grad[2:static_node_num, :] * 0.01 * 3
            # samples[:static_node_num, 0] = np.clip(samples[:static_node_num, 0], lb[0], ub[0])
            # samples[:static_node_num, 1] = np.clip(samples[:static_node_num, 1], lb[1], ub[1])
            samples[:static_node_num] = np.maximum(samples[:static_node_num], hand.nn.hand_bound[0, :])  # seems have bug here, use np.clip
            samples[:static_node_num] = np.minimum(samples[:static_node_num], hand.nn.hand_bound[1, :])
        # print('time cost for update samples', time.time() - t0)

    # update graph
    # t0 = time.time()
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
    pairs = hand.collision_hand_obj_SCA(q=None, sample=edge_sample_num)  # do the hand-obj collision and SCA
    start_end_p = []
    for i in range(len(s2)):
        for j in range(k0):
            start_end_p.append(s1[i] + s2[i][j])
    d_star.graph.E = dict(zip(start_end_p, pairs))
    # print('knn + update graph', time.time() - t0)
    # update cost

    result = d_star.ComputePath()

    path = d_star.extract_path(max_nums=max_nums)
    # if result:
    #     path = d_star.extract_path(max_nums=max_nums)
    # else:
    #     path = []
    # print('step', step, len(path), '   find path cost:', time.time() - t0)
    time_cost = time.time() - t0
    if np.linalg.norm(np.array(path[-1]) - np.array(s_goal)) < 1e-5:
        a = 1
        print('Success, time all', time_cost)
        metrics['success'].append(1)

        if optimize:
            s1 = path
            s2 = [path[:i] + path[i+1:] for i in range(len(path))]
            graph_2D_opt = graph(dim=dim)
            graph_2D_opt.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
            d_star_opt = DStar(s_start, s_goal, graph_2D_opt, "euclidean")
            samples_opt = np.array(s1)
            start_p_opt = np.repeat(samples_opt, repeats=len(s1)-1, axis=0)
            edge_samples_ = np.linspace(start_p_opt, np.vstack(s2),
                                        edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
            edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

            pairs = hand.collision_hand_obj_SCA(sample=edge_sample_num, safety_dis=safety_dis / 5)
            start_end_p = []
            for i in range(len(s1)):
                for j in range(len(s1)-1):
                    start_end_p.append(s1[i] + s2[i][j])
            d_star_opt.graph.E = dict(zip(start_end_p, pairs))
            # print('knn + update graph', time.time() - t0)
            # update cost
            result_opt = d_star_opt.ComputePath()

            path_opt = d_star_opt.extract_path(max_nums=max_nums)
            time_cost = time.time() - t0

            path = path_opt

        metrics['time_cost'].append(time_cost)
        path_len = 0
        for j in range(len(path) - 1):
            path_len += np.linalg.norm(np.array(path[j]) - np.array(path[j+1]))
        metrics['path_len'].append(path_len)

    else:
        metrics['success'].append(0)
        print('No feasible path')
    # figures

    figure = False
    if figure:
        save_path = 'figures/PRM_static/'
        os.makedirs(save_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        output = hand.get_full_map_dis()
        con = plt.contourf(hand.x_grid, hand.y_grid, output, np.linspace(-1, 1, 11), cmap='PiYG')

        plt.contour(con, levels=[0], colors=('k',), linestyles=('-',),
                    linewidths=(2,))  # boundary of the obs in C space
        plt.title(label='Isolines and PRM')
        cax = plt.axes([0.95, 0.1, 0.05, 0.8])
        plt.colorbar(con, cax=cax)
        plt.title(label='Dis')

        ax.set_xlabel('$q_1$' + ' (rad)')
        ax.set_ylabel('$q_2$' + ' (rad)')
        for k, v in d_star.graph.edges.items():
            for vi in v:
                if d_star.graph.E[k + vi]:
                    ax.plot([k[0], vi[0]], [k[1], vi[1]], color='gray')
        ax.scatter(samples[:, 0], samples[:, 1], c='k', zorder=100)  # all nodes of PRM
        ax.scatter(d_star.s_start[0], d_star.s_start[1], c='b', zorder=100)
        ax.scatter(s_goal[0], s_goal[1], c='r', zorder=100)
        # ax.scatter(s_robot[0], s_robot[1], c='k', zorder=100)
        # plot feasible path

        if 1000 > len(path) > 1:
            if optimize:
                path = path_opt
            for i in range(len(path) - 1):
                ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='r')

        fig.savefig(save_path + '16D_PRM_static_' + str(step) + '.png', format='png', bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300)
        plt.close()

    step += 1

print("success rate:", sum(metrics['success'])/ len(metrics['success']))
print('time cost:', np.mean(metrics['time_cost']), np.std(metrics['time_cost']) )
print('Path len:', np.mean(metrics['path_len']), np.std(metrics['path_len']) )



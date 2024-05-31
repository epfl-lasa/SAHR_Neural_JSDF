import numpy as np
import time
import torch
from rtree import index
import time

from PRM_tools import collision_check, graph
import matplotlib.pyplot as plt

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
n = 300

from D_star_lite import DStar

# D star
pairs_last = []
metrics = {'success':[], 'time_cost':[], 'path_len':[] }
step_max = 100

for step in range(step_max):
    samples = np.random.uniform(lb, ub, size=(n, dim))

    samples_col_bool = collision.obstacle_free(samples)

    # samples_free = samples[samples_col_bool, :]  # use only collision-free samples
    samples_free = samples  # use all samples
    # print('ratio of collision-free samples', samples_free.shape[0] / n)

    # build graph

    graph_2D = graph(dim=dim)
    sample_free_tuple = list(map(tuple, samples_free))
    s_start = (0, 0)  # starting location
    s_goal = (1.25, 0.25)  # goal location
    sample_free_tuple.append(s_start)
    sample_free_tuple.append(s_goal)

    [graph_2D.add_vertex(sample) for sample in sample_free_tuple]

    # k-NN nearest
    t0 = time.time()
    k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 5
    edges = {}  # to be checked if collision-free
    edge_sample_num = 10

    # collision check for edges
    t0 = time.time()

    start_p = []
    end_p = []
    start_end_p = []
    for i in range(len(sample_free_tuple)):
        q = sample_free_tuple[i]
        if q in [s_start, s_goal]:
            k = k0 + 20
        else:
            k = k0
        near_q = graph_2D.nearby(q, k)
        near_q_list = []
        for near_j in near_q:
            if near_j == q:
                continue
            start_p.append(q)
            start_p.append(near_j)
            end_p.append(near_j)
            end_p.append(q)
            near_q_list.append(near_j)
            start_end_p.append(q + near_j)
            start_end_p.append(near_j + q)
            if near_j in edges:
                if q not in edges[near_j]:
                    edges[near_j].append(q)
            else:
                edges[near_j] = [q]
        if q in edges:
            edges[q] += near_q_list
        else:
            edges[q] = near_q_list

        # assert len(near_q_list) == k
        assert q not in edges[q]

    edge_samples_ = np.linspace(np.vstack(start_p), np.vstack(end_p),
                                edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
    edge_samples = np.vstack(edge_samples_)
    print(edge_samples.shape)
    # print('K-NN time:', time.time() - t0)

    # step = 0

    graph_2D.edges = edges  # {v1: [a1,a2,a3], v2; [...],...}

    # x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
    x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01], ])  # this one is consistent with the paper. narrow passage
    # x_obj += np.array([1.8e-2 * np.sin(step / 20), 0, 0])
    x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
    collision.x_obj_gpu = x_obj_gpu

    t0 = time.time()
    pairs = collision.get_dis(edge_samples, gradient=False, sample=edge_sample_num)
    # print('edge collision check time:', time.time() - t0)
    d_star = DStar(s_start, s_goal, graph_2D, "euclidean")

    d_star.graph.E = dict(zip(start_end_p, pairs))  # {v1 + v2 : False/True, ...}

    if len(pairs_last) != 0:
        diff = np.logical_xor(pairs, pairs_last)  # True means the state is changed
        # store the edges that need to update
        edges2update = [start_end_p[i] for i in range(len(start_end_p)) if diff[i]]
        print('edge num to be updated', len(edges2update))
        d_star.update_cost(edges2update)
    pairs_last = pairs

    result = d_star.ComputePath()
    if result:
        path = d_star.extract_path()
    else:
        path = []
    print('step', step, len(path), '   find path cost:', time.time() - t0)

    if 1 < len(path) < 1000:
        a = 1
        pass
    # figures
        if np.linalg.norm(np.array(path[-1]) - np.array(s_goal)) < 1e-5:
            a = 1
            time_cost = time.time() - t0
            print('Success, time all', time_cost)
            metrics['success'].append(1)
            metrics['time_cost'].append(time_cost)
            path_len = 0
            for j in range(len(path) - 1):
                path_len += np.linalg.norm(np.array(path[j]) - np.array(path[j + 1]))
            metrics['path_len'].append(path_len)

    figure = False
    if figure:
        save_path = 'figures/'
        fig, ax = plt.subplots(figsize=(8, 6))
        output = collision.get_full_map_dis()
        con = plt.contourf(collision.x_grid, collision.y_grid, output, np.linspace(-1, 1, 11), cmap='PiYG')

        plt.contour(con, levels=[0], colors=('k',), linestyles=('-',),
                    linewidths=(2,))  # boundary of the obs in C space
        plt.title(label='Isolines and PRM')
        cax = plt.axes([0.95, 0.1, 0.05, 0.8])
        plt.colorbar(con, cax=cax)
        plt.title(label='Dis')

        ax.set_xlabel('$q_1$' + ' (rad)')
        ax.set_ylabel('$q_2$' + ' (rad)')
        for k, v in edges.items():
            for vi in v:
                if d_star.graph.E[k + vi]:
                    ax.plot([k[0], vi[0]], [k[1], vi[1]], color='gray')

        ax.scatter(s_start[0], s_start[1], c='b', zorder=100)
        ax.scatter(s_goal[0], s_goal[1], c='r', zorder=100)
        # plot feasible path
        if len(path):
            for i in range(len(path) - 1):
                ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='r')

        fig.savefig(save_path + 'moving_2D_02_' + str(step) + '.png', format='png', bbox_inches='tight', pad_inches=0.0,
                    dpi=300)
        plt.close()


print("success rate:", sum(metrics['success'])/ step_max)
print('time cost:', np.mean(metrics['time_cost']), np.std(metrics['time_cost']) )
print('Path len:', np.mean(metrics['path_len']), np.std(metrics['path_len']) )

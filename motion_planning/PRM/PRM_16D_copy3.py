import numpy as np
import time
import torch
from rtree import index
import time

from PRM_tools import collision_check, graph
import matplotlib.pyplot as plt

# initialize the env

g = [0, 1, 2, 3, 4]  # palm, index/middle/ring/thumb finger
use_cuda = True

x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])

hand = collision_check(x_obj, use_cuda=use_cuda)

# generate samples
dim = 16
n = 200
success = 0
trial_max = 100
# np.random.seed(0)
metrics = {'success': [], 'time_cost': [], 'path_len': []}
optimize = True
dynamic = True
safety_dis = 1e-4
for ii in range(trial_max):
    t0 = time.time()
    samples = np.random.uniform(hand.nn.hand_bound[0, :], hand.nn.hand_bound[1, :], size=(n, dim))
    if dynamic:
        t0 = time.time()
        static_node_num = 50
        for i in range(3):  # update samples to make them closer to the safety_dis
            dis, grad = hand.get_dis(samples, gradient=True, real_dis=True, dx=True)
            # dx = grad[:, 4:]
            # x_change = np.array([1.8e-2 * np.sin(step / 20), 0, 0]) - np.array([1.8e-2 * np.sin((step-1) / 20), 0, 0])

            # samples[2:static_node_num, :] = samples[2:static_node_num, :] - (dis[2:static_node_num].reshape(-1, 1) - safety_dis * 2 - dx[2:static_node_num] @ x_change.reshape(-1,1)) / grad[2:static_node_num, 1:3] * 0.01 * 3
            samples[2:static_node_num, :] = samples[2:static_node_num, :] - (dis[2:static_node_num].reshape(-1, 1) - safety_dis * 2) / grad[2:static_node_num, :] * 0.01 * 3
            # samples[:static_node_num, 0] = np.clip(samples[:static_node_num, 0], lb[0], ub[0])
            # samples[:static_node_num, 1] = np.clip(samples[:static_node_num, 1], lb[1], ub[1])
            samples[:static_node_num] = np.maximum(samples[:static_node_num], hand.nn.hand_bound[0, :])
            samples[:static_node_num] = np.minimum(samples[:static_node_num], hand.nn.hand_bound[1, :])
    samples_col_bool = hand.obstacle_free(samples)

    # samples_free = samples[samples_col_bool, :]  # use only collision-free samples
    samples_free = samples  # use all samples
    # print('ratio of collision-free samples', samples_free.shape[0] / n)

    # build graph

    graph_D = graph(dim=dim)
    sample_free_tuple = list(map(tuple, samples_free))
    q0 = np.zeros(16)
    q0[12] = 0.5
    s_start = tuple(q0)  # starting location

    s_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
              -0.06984199999999996, 1.3148996000000002, 1.27591, 0.43,
              0.0, 1.3293476000000002, 1.2510544000000001, 0.55,
              1.1357499, 0.9659528, 1.5200892, 0.6767379)  # goal location

    sample_free_tuple.append(s_start)
    sample_free_tuple.append(s_goal)

    [graph_D.add_vertex(sample) for sample in sample_free_tuple]

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
            # k = k0 + 0
        else:
            k = k0
        near_q = graph_D.nearby(q, k)
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
    # print(edge_samples.shape)
    # print('K-NN time:', time.time() - t0)

    hand.q_gpu = torch.Tensor(edge_samples).to('cuda:0') if use_cuda else torch.Tensor(edge_samples)

    # step = 0
    step_max = 1
    graph_D.edges = edges  # {v1: [a1,a2,a3], v2; [...],...}

    from D_star_lite import DStar

    # D star
    d_star = DStar(s_start, s_goal, graph_D, "euclidean")
    pairs_last = []
    # s_v = 0.02
    # s_robot = np.array(s_start)
    t_list = {'collision': [], 'update_vertices': [], 'd_star': []}

    for step in range(step_max):
        # x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
        x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])
        # x_obj += np.array([1.8e-2 * np.sin(step / 20), 0, 0])
        x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
        hand.x_obj_gpu = x_obj_gpu


        pairs = hand.collision_hand_obj_SCA(q=None, sample=edge_sample_num, safety_dis=safety_dis)
        t_list['collision'].append(time.time() - t0)
        # print('edge collision check time:', time.time() - t0)

        d_star.graph.E = dict(zip(start_end_p, pairs))  # {v1 + v2 : False/True, ...}

        if len(pairs_last) != 0:
            diff = np.logical_xor(pairs, pairs_last)  # True means the state is changed
            # store the edges that need to update
            edges2update = [start_end_p[i] for i in range(len(start_end_p)) if diff[i]]
            # print('edge num to be updated', len(edges2update))
            d_star.update_cost(edges2update)
        pairs_last = np.copy(pairs)
        t_list['update_vertices'].append(time.time() - t0)

        # t0 = time.time()
        result = d_star.ComputePath()
        if result:
            path = d_star.extract_path()
            # direction = np.array(path[1]) - s_robot
            # s_robot = s_robot + s_v * direction / (np.linalg.norm(direction) + 8e-3)
            # dis = np.linalg.norm(s_robot - np.array(path[1]))
            # if dis < 9e-3:
            #     d_star.s_start = path[1]
            #     d_star.km += d_star.c[path[0] + path[1]]
            #     print('robot reached the next node. Step', step)
            #     path.pop(0)
            #     if path[1] == s_goal:
            #         print('Reach the goal')
            #         break
        else:
            path = []
        # print('step', step, len(path), '   find path cost:', time.time() - t0)

        t_list['d_star'].append(time.time() - t0)
        time_cost = time.time() - t0
        if 1 < len(path) < 1000:
            a = 1
            pass

            if np.linalg.norm(np.array(path[-1]) - np.array(s_goal)) < 1e-5:
                if optimize:
                    s1 = path
                    s2 = [path[:i] + path[i + 1:] for i in range(len(path))]
                    graph_2D_opt = graph(dim=dim)
                    graph_2D_opt.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
                    d_star_opt = DStar(s_start, s_goal, graph_2D_opt, "euclidean")
                    samples_opt = np.array(s1)
                    start_p_opt = np.repeat(samples_opt, repeats=len(s1) - 1, axis=0)
                    edge_samples_ = np.linspace(start_p_opt, np.vstack(s2),
                                                edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
                    edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)
                    hand.q_gpu = torch.Tensor(edge_samples).to('cuda:0') if use_cuda else torch.Tensor(edge_samples)

                    pairs = hand.collision_hand_obj_SCA(sample=edge_sample_num, safety_dis=safety_dis / 5)
                    start_end_p = []
                    for i in range(len(s1)):
                        for j in range(len(s1) - 1):
                            start_end_p.append(s1[i] + s2[i][j])
                    d_star_opt.graph.E = dict(zip(start_end_p, pairs))
                    # print('knn + update graph', time.time() - t0)
                    # update cost
                    result_opt = d_star_opt.ComputePath()

                    path_opt = d_star_opt.extract_path(max_nums=1000)
                    time_cost = time.time() - t0

                    path = path_opt
                a = 1
                success += 1
                print('Success, for loop', ii)
                metrics['success'].append(1)
                metrics['time_cost'].append(time_cost)
                path_len = 0
                for j in range(len(path) - 1):
                    path_len += np.linalg.norm(np.array(path[j]) - np.array(path[j + 1]))
                metrics['path_len'].append(path_len)
        # figures
        figure = False
        if figure:
            save_path = 'figures/'
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

            fig.savefig(save_path + 'moving_2D_01_' + str(step) + '.png', format='png', bbox_inches='tight',
                        pad_inches=0.0,
                        dpi=300)
            plt.close()

print("success rate:", sum(metrics['success']) / trial_max)
print('time cost:', np.mean(metrics['time_cost']), np.std(metrics['time_cost']))
print('Path len:', np.mean(metrics['path_len']), np.std(metrics['path_len']))

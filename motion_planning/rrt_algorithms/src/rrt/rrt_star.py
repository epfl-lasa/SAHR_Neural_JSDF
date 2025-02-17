# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import time
from operator import itemgetter

import numpy as np
from src.rrt.heuristics import cost_to_go
from src.rrt.heuristics import segment_cost, path_cost
from src.rrt.rrt import RRT
import random
import matplotlib.pyplot as plt
import torch


class RRTStar(RRT):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01, rewire_count=None, DS_step=None, vel=None,
                 get_dis=None, figures=None, get_full_dis=None, parallel=False, dim=None):
        """
        RRT* Search
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param rewire_count: number of nearby vertices to rewire
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc, DS_step=DS_step, vel=vel, get_dis=get_dis, dim=dim)
        self.x_obj_gpu = None
        self.x_rewire = []
        self.rewire_count = rewire_count if rewire_count is not None else 0
        self.c_best = float('inf')  # length of best solution thus far
        self.DS_step = DS_step
        self.get_dis = get_dis
        self.max_samples = max_samples
        self.dim = dim
        # self.X = X
        # self.X.
        self.x_goal = x_goal
        if figures is not None:
            self.x_grid = figures[0]
            self.y_grid = figures[1]
            self.level = figures[2]
        # # if get_full_dis is not None:
        self.get_full_dis = get_full_dis
        self.parallel = parallel

    def get_nearby_vertices(self, tree, x_init, x_new):
        """
        Get nearby vertices to new vertex and their associated path costs from the root of tree
        as if new vertex is connected to each one separately.

        :param tree: tree in which to search
        :param x_init: starting vertex used to calculate path cost
        :param x_new: vertex around which to find nearby vertices
        :return: list of nearby vertices and their costs, sorted in ascending order by cost
        # path_cost woule be inf if the node collides with obstacle (not in E)
        """
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        # X_near_list = list(X_near)
        # if self.x_goal not in X_near_list:
        #     X_near_list.append(self.x_goal)  # the new point might be a better parent for x_goal

        L_near = [(path_cost(self.trees[tree].E, x_init, x_near) + segment_cost(x_near, x_new), x_near) for
                  x_near in X_near]
        # noinspection PyTypeChecker
        L_near.sort(key=itemgetter(0))

        return L_near

    def rewire(self, tree, x_new, L_near):
        """
        Rewire tree to shorten edges if possible
        Only rewires vertices according to rewire count
        :param tree: int, tree to rewire
        :param x_new: tuple, newly added vertex, this might be a parent for near vertexes
        :param L_near: list of nearby vertices used to rewire
        :return:
        # segment_cost: Euclidean distance
        # self.trees[tree].E[child] = parent
        """
        for c_near, x_near in L_near:
            tent_cost = path_cost(self.trees[tree].E, self.x_init, x_new) + segment_cost(x_new, x_near)
            if tent_cost == np.inf:
                continue
            curr_cost = path_cost(self.trees[tree].E, self.x_init, x_near)

            if tent_cost < curr_cost and self.X.collision_free(x_near, x_new, self.r):
                if x_near in self.trees[0].E:
                    self.trees[0].E_samples.pop(x_near + self.trees[0].E[x_near])
                self.trees[tree].E[x_near] = x_new  # use x_new for new parent of x_near
                self.trees[tree].E_samples[x_near + x_new] = np.linspace(x_near, x_new, 50)

    def connect_shortest_valid(self, tree, x_new, L_near):
        """
        Connect to nearest vertex that has an unobstructed path
        :param tree: int, tree being added to
        :param x_new: tuple, vertex being added
        :param L_near: list of nearby vertices, and their costs, sorted in ascending order by cost
         ## cost_to_go, the Euclidean distance to goal
         ## connect_to_point, True if able to add edge, False if prohibited by an obstacle
         # notes that c_best is always inf.
        """
        # check nearby vertices for total cost and connect shortest valid edge
        for c_near, x_near in L_near:
            if x_near != self.x_goal:
                if c_near + cost_to_go(x_near, self.x_goal) < self.c_best and self.connect_to_point(tree, x_near,
                                                                                                    x_new):  # x_near parent
                    break
        # self.trees[0].E_dis_grad[x_new] = x_near

    def current_rewire_count(self, tree):
        """
        Return rewire count
        :param tree: tree being rewired
        :return: rewire count
        """
        # if no rewire count specified, set rewire count to be all vertices
        if self.rewire_count is None:
            return self.trees[tree].V_count

        # max valid rewire count
        return min(self.trees[tree].V_count, self.rewire_count)

    def rrt_star(self, variable_step=False, figure=True):
        """
        Based on algorithm found in: Incremental Sampling-based Algorithms for Optimal Motion Planning (RRT*)
        http://roboticsproceedings.org/rss06/p34.pdf
        :return: set of Vertices; Edges in form: vertex: [neighbor_1, neighbor_2, ...]
        """
        self.add_vertex(0, self.x_init)
        # self.add_edge(0, self.x_init, None)
        use_cuda = False  # todo, put into self.
        # print(self.trees[0].V)
        # print(list(self.trees[0].V.intersection((-2, -1.0, 2.0, 2.0))))

        q = self.Q[0]
        step = 0

        radius = 0.01 * 1.5
        # print(self.get_dis((1.0, -0.2)))
        T_o = {}
        offset_x_new = False  # use the DS integral to update x_rand
        moving = True
        kk = 0
        solution = [False, None, 0]
        path_valid = False
        while True:
            kk += 1
            if kk >= self.max_samples or step >400:
                break
                # tmp=1
            # print(k)
            t0 = time.time()
            # change the position of obstacle
            # x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
            # x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01],
            #                   ])
            x_obj = np.array([[7.90000000e-02, 0, 1.97000000e-01], ]) # for 2D case
            # x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187]])
            # x_obj = np.array([[0.08, 0, 0.187], [0.08, -0.06, 0.187], [0.08, 0.06, 0.187], [0.11, 0.03, 0.087]]) # 16D final one
            if moving:
                x_obj += np.array([1.8e-2 * np.sin(step/20), 0, 0])
            x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if use_cuda else torch.Tensor(x_obj)
            self.X.x_obj = x_obj_gpu
            if moving:
                T_f = {k: v for k, v in self.trees[0].E_dis_grad.items() if v[0] > 0}  # collision free nodes
                T_o_last = T_o
                T_o = {k: v for k, v in self.trees[0].E_dis_grad.items() if v[0] <= 0}  # collision nodes
                if T_o:
                    # print("collision node", T_o)
                    for i in T_o_last:
                        if i not in T_o:
                            self.x_rewire.append(i)

                all_samples = list(self.trees[0].E_samples.values())
                if len(all_samples):
                    pairs = self.get_dis(np.vstack(all_samples), gradient=False, x_obj=x_obj_gpu,
                                         sample=50)  # input all edges
                    tmp = dict(zip(list(self.trees[0].E_samples), pairs))  # true means the edge has no collision
                    # remove the edges who have collision as a child., also remove edges whose parent is the
                    # collision node?
                    E_new = {}
                    for k, v in self.trees[0].E.items():
                        if tmp[k + v]:
                            E_new[k] = v
                        else:
                            self.trees[0].E_samples.pop(k + v)
                            self.x_rewire.append(k)
                            self.x_rewire.append(v)

                    # E_new = {k: v for k, v in self.trees[0].E.items() if k in T_f and (v in T_f or v is None) and tmp[k+v]}  # all edges are connected by collision-free nodes
                    self.trees[0].E = E_new
                # also remove edges whose parent is the collision node?
                self.x_rewire.append(self.x_goal)

            if solution[0]:
                sol = np.array(solution[1])
                edge_samples_ = np.linspace(sol[:-1, :], sol[1:, :],
                                            50, axis=1)  # (n*k, edge_sample_num, dim)
                edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)
                path_check = self.get_dis(edge_samples, gradient=False, x_obj=x_obj_gpu,
                                 sample = 150 )
                if np.all(path_check):
                    pass  # the path is still valid
                    path_valid = True
                else:
                    # sampling more points
                    path_valid = False

            # if not path_valid or self.DS_step is None:
            if not path_valid:
                # pairs = self.get_dis(np.vstack(all_samples), gradient=False, x_obj=x_obj_gpu,
                #                      sample=50)
                # tmp = self.get_dis(np.array(self.x_init), gradient=False,
                #              x_obj=x_obj_gpu)
                # print(self.X.collision_free((0.8,-0.1), self.x_goal, self.r))

                # update the dis and grad for all nodes.

                if variable_step:
                    dis, grad = self.get_dis(np.vstack(self.trees[0].V_all), gradient=True,
                                             x_obj=x_obj_gpu)  # input all nodes
                    dis_grad = np.concatenate([dis.reshape(-1, 1), grad], axis=1)  # (n x 3)
                    #  Updates if 'a' exists, else adds 'a' into the dict
                    # self.trees[0].E_dis_grad.update(dict(zip(self.trees[0].V_all, dis_grad)))
                    self.trees[0].E_dis_grad.update(dict(zip(self.trees[0].V_all, dis_grad)))
                    # update nodes who have collision or not
                      # add to rewire list because the collision state has changed
                            # x_rewire = {k: v for k, v in self.trees[0].E.items() if v in T_o} # the nodes who have collision
                            # parents


                    # only delete edge who
                    # for k in self.trees[0].E:
                    #     if k in T_o:
                    #         self.trees[0].E.pop(k)
                    # self.trees[0].E = {k: v for k, v in self.trees[0].E.items() if k not in s}

                # for q in self.Q:  # iterate over different edge lengths
                #     for i in range(int(q[1])):  # iterate over number of edges of given length to add
                x_new, x_nearest = self.new_and_near(0, q, variable_step=variable_step, offset_x_new=offset_x_new)

                if x_new is None:
                    continue

                # get nearby vertices and cost-to-come
                L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                # check nearby vertices for total cost and connect shortest valid edge
                self.connect_shortest_valid(0, x_new, L_near)

                if x_new in self.trees[0].E:
                    # rewire tree
                    self.rewire(0, x_new, L_near)
                # for nodes who have collision parents
                # t0 = time.time()
            if len(self.x_rewire) != 0:
                for x in list(set(self.x_rewire)):  # use set to remove repeated nodes
                    X_near = self.nearby(0, x, 20)
                    for x_near in X_near:
                        tent_cost = path_cost(self.trees[0].E, self.x_init, x_near) + segment_cost(x_near, x)
                        if tent_cost == np.inf or x_near == self.x_goal:
                            continue
                        curr_cost = path_cost(self.trees[0].E, self.x_init, x)
                        if tent_cost < curr_cost and self.X.collision_free(x, x_near, self.r):
                            if x in self.trees[0].E:
                                self.trees[0].E_samples.pop(x + self.trees[0].E[x])
                            if x != self.x_goal:
                                self.trees[0].E[x] = x_near  # use x_new for new parent of x_near
                            self.trees[0].E_samples[x + x_near] = np.linspace(x, x_near, 50)
                self.x_rewire = []

            if self.DS_step is None:
                solution = self.check_solution()
            else:
                # try to check if the DS lead current point to the goal

                # t0 = time.time()
                # x_goal_near = self.nearby(0, x_new, self.current_rewire_count(0))
                # x_test = [x for x in x_goal_near]
                # solution = self.solution_DS_check_all(np.vstack(x_test))
                # solution = self.solution_DS_check_all(np.vstack(T_f))
                if len(T_f) > 0:
                    solution = self.solution_DS_check_all(T_f)
                if self.x_goal in self.trees[0].E:
                    # pass
                    aaa = 1
                # if x_new in self.trees[0].E:
                # solution = self.solution_DS_check_all_(x_new)
                # path_len = np.inf
                # best_solution = [False,None,None, None]
                # for xx in T_f:
                #     solution = self.solution_DS_check_all_(xx)
                #     if solution[0]:
                #         cost = 0
                #         for j in range(len(solution[1]) - 1):
                #             cost += np.linalg.norm(np.array(solution[1][j]) - np.array(solution[1][j + 1]))
                #         if cost < path_len:
                #             path_len = cost
                #             best_solution = solution
                # if best_solution[0]:
                #     solution = best_solution

                if solution[0]:
                    self.x_rewire += solution[4]
                # if not solution[0]:
                #     solution = self.check_solution()
                # solution = self.solution_DS_check(x_new)
                # solution = self.solution_DS_check_all(np.vstack(x_test))

            # print('solution check time cost:', time.time() - t0)
            # print('rrt time cost:', time.time() - t0, " node number", len(self.trees[0].V_all))

            # plot figures
            # if figure and solution[0]:
            if figure:
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.size"] = "8"
                from matplotlib import rc
                rc('text', usetex=True)
                fig_width = 90 / 25.4
                # path = 'output/visualizations/2D_static_compare/DS_' + str(step)
                # path = 'output/visualizations/2D_static_compare/DS_final'
                path = 'output/visualizations/2D_moving_DS/DS_final/DS_' + str(step)
                # path = 'output/visualizations/2D_moving_DS/DS_final/DS_'
                # path = 'output/visualizations/2D_moving_DS/RRT_' + str(step)
                # path = 'output/visualizations/2D_DS_check/' + str(step)

                fig, ax = plt.subplots(figsize=(fig_width, fig_width / 4 * 3))
                ##################
                output = self.get_full_dis(x_obj_gpu)

                con = plt.contourf(self.x_grid, self.y_grid, output, self.level, cmap='PiYG')

                plt.contour(con, levels=[0], colors=('k',), linestyles=('-',),
                            linewidths=(2,))  # boundary of the obs in C space
                # plt.title(label='Isolines and RRT tree largeness, node number:' + str(len(self.trees[0].V_all)))
                cax = plt.axes([0.83, 0.1, 0.05, 0.8])
                plt.colorbar(con, cax=cax)
                plt.title(label='Dis')
###################################

                ax.set_xlabel('Middle $q_2$' + ' (rad)')
                ax.set_ylabel('Middle $q_3$' + ' (rad)')
                # ax.set_xlim(nn.hand_bound[:, 1])
                # ax.set_ylim(nn.hand_bound[:, 2])
                ax.set_aspect('equal', adjustable='box')
                if self.dim == 2:
                    pass
                    for k, v in self.trees[0].E.items():
                        if v is not None:
                            if k == x_new:
                                ax.plot([k[1], v[1]], [k[2], v[2]], color='k')
                            else:
                                ax.plot([k[1], v[1]], [k[2], v[2]], color='k')
                    if solution[0]:  # plot the solution path
                        for i in range(len(solution[1]) - 1):
                            ax.plot([solution[1][i][1], solution[1][i + 1][1]],
                                    [solution[1][i][2], solution[1][i + 1][2]],
                                    color='r')
                    ax.scatter(self.x_init[1], self.x_init[2], c='b', zorder=100)
                    ax.scatter(self.x_goal[1], self.x_goal[2], c='r', zorder=100)
                    ax.plot([x_new[1], self.x_goal[1]], [x_new[2], self.x_goal[2]])

                else:
                    for k, v in self.trees[0].E.items():
                        if v is not None:
                            if k == x_new:
                                ax.plot([k[0], v[0]], [k[1], v[1]], color='k')
                            else:
                                ax.plot([k[0], v[0]], [k[1], v[1]], color='k')
                    if solution[0]:  # plot the solution path
                        c_path = 'r'
                        first_legend = True
                        for i in range(len(solution[1]) - 1):
                            legend_label = 'RRT*'
                            if self.DS_step is not None:
                                if i > solution[2]-1:
                                    c_path = 'b'
                                    legend_label = 'DS'
                                    if i == len(solution[1]) - solution[2]:
                                        first_legend = True

                            if first_legend:
                                ax.plot([solution[1][i][0], solution[1][i + 1][0]],
                                        [solution[1][i][1], solution[1][i + 1][1]],
                                        color=c_path, label=legend_label)
                                first_legend=False
                            else:
                                ax.plot([solution[1][i][0], solution[1][i + 1][0]],
                                        [solution[1][i][1], solution[1][i + 1][1]],
                                        color=c_path)
                    ax.scatter(self.x_init[0], self.x_init[1], c='b', zorder=100)
                    ax.scatter(self.x_goal[0], self.x_goal[1], c='r', zorder=100)
                    if solution[0]:
                        for i in range(solution[3].shape[0]-1):
                            ax.plot([solution[3][i, 0], solution[3][i + 1, 0]],
                                    [solution[3][i, 1], solution[3][i + 1, 1]],
                                    color='b')
                if solution[0]:
                    ax.legend(loc = 'upper right')
                # fig.savefig(path + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0.0)
                fig.savefig(path + '.png', format='png', bbox_inches='tight', pad_inches=0.0, dpi=300)
                plt.close()


            step += 1

            if solution[0]:
                aaa = 3

                # print(self.trees[0].V)

















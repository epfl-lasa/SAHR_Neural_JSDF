import random

import numpy as np

from src.rrt.tree import Tree
from src.utilities.geometry import steer
from src.rrt.heuristics import segment_cost, path_cost
import copy


class RRTBase(object):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01, DS_step=None, vel=None, get_dis=None,
                 get_full_dis=None, dim=None):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q
        self.r = r
        self.prc = prc
        self.x_init = x_init
        self.x_goal = x_goal
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree
        self.dim = dim

        self.DS_step = DS_step
        self.vel = vel
        self.get_dis = get_dis
        self.get_full_dis = get_full_dis

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(Tree(self.X))

    def add_vertex(self, tree, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """
        self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].V_count += 1  # increment number of vertices in tree

        self.trees[tree].V_all.append(v)  # store all vertices
        self.samples_taken += 1  # increment number of samples taken

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        assert len(child) != self.x_goal
        # assert len(parent) == 2
        self.trees[tree].E[child] = parent
        if parent is not None:
            self.trees[tree].E_samples[child + parent] = np.linspace(child, parent, 50)  # samples

    def nearby(self, tree, x, n):
        """
        Return nearby vertices
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :param n: int, max number of neighbors to return
        :return: list of nearby vertices
        """
        return self.trees[tree].V.nearest(x, num_results=n, objects="raw")

    def get_nearest(self, tree, x, n=1):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        :return: tuple, nearest vertex to x
        """
        return next(self.nearby(tree, x, n))

    def new_and_near(self, tree, q, variable_step=False, offset_x_new=False):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
        """
        n = 1
        while 1:
            x_rand = self.X.sample_free()
            if self.dim == 8:
                x_rand[4:12] = np.zeros(8)
            if self.dim == 4:
                x_rand[4:] = np.zeros(12)
                x_rand[12] = 0.5
            if self.dim ==2:
                x_rand[0] = 0
                x_rand[3:] = np.zeros(13)
                x_rand[12] = 0.5
            x_nearest = self.get_nearest(tree, x_rand, n=n)  # x_nearest might have collision
            # if n > 1:
            #     x_nearest = x_nearest[-1]
            if x_nearest in self.trees[0].E or x_nearest in self.trees[
                0].E.values() or x_nearest == self.x_init:  # could also connect one as parent
                # if so,means that x_nearest has no collision
                break
            else:
                n += 1
        if variable_step:
            # pass
            # if x_nearest
            dis_grad = self.trees[0].E_dis_grad[x_nearest]

            dis = dis_grad[0:1]
            grad = dis_grad[1:]
            # grad[4:12] = np.zeros(8)
            if dis < 0:
                return None, None
            start, end = np.array(x_nearest), np.array(x_rand)
            v = end - start
            u = v / np.linalg.norm(v)

            dot_tmp = np.dot(u, grad)
            if dot_tmp > 0:
                step_max = q[0]
            else:
                step_max = - dis / np.dot(u, grad) * 0.7
                # step_max = np.clip(step_max, q[0], 1)
                step_max = np.clip(step_max, q[0] / 5, q[0])
                # print(step_max)
            x_new = self.bound_point(steer(x_nearest, x_rand, step_max))
            # variable_step_size  =

        else:
            # x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
            # np.random.uniform(q[0] / 5, q[0])
            x_new = self.bound_point(steer(x_nearest, x_rand, q[0] / 3))
            # dis = self.get_dis(x_new)
        # check if new point is in X_free and not already in V
        # dis, grad = self.get_dis(x_new, gradient=True)

        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new):
            # if not self.trees[0].V.count(x_new) == 0 or not dis > 0:
            return None, None
        self.samples_taken += 1
        # if variable_step:
        #     self.trees[0].E_dis_grad[x_new] = np.concatenate([dis, grad])

        # if offset_x_new and random.random() < self.prc:
        if offset_x_new:
            K = 5
            dt = 0.01
            # x_DS = np.zeros([x.shape[0], 2, step + 1])
            # x_DS[:, :, 0] = np.copy(x)
            x = np.array(x_new)
            for i in range(K):
                dq = self.vel(x, self.X.x_obj)
                x = x + dq * dt
                x[4:12] = np.zeros(8)
            x_new = tuple(x)

        return x_new, x_nearest

    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex, parent
        :param x_b: tuple, vertex, child
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        assert  x_b != self.x_goal
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        # if len(x_nearest) > 1:
        #     x_nearest = x_nearest[0]
        # tmp1 = self.x_goal in self.trees[tree].E
        # if self.x_goal in self.trees[tree].E and x_nearest == self.trees[tree].E[self.x_goal]:
        if self.x_goal in self.trees[tree].E:
            # tree is already connected to goal using nearest vertex
            return True
        tmp = self.X.collision_free(x_nearest, self.x_goal, self.r)
        if tmp:  # check if obstacle-free
            return True
        return False

    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.can_connect_to_goal(0):  # should not use this for generate path
            # print("goal has a parent")
            # self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        # print("Could not connect to goal")
        return None

    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        (does not check if this should be possible, for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest
        self.trees[tree].V_all.append(self.x_goal)
        self.trees[tree].E_samples[self.x_goal + x_nearest] = np.linspace(self.x_goal, x_nearest, 50)  # samples

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while 1:  # key error problem.
            if not (current in self.trees[tree].E):
                self.x_rewire.append(current)  # current point have no parent
                return None
            else:
                path.append(self.trees[tree].E[current])
                current = self.trees[tree].E[current]
            if current in self.trees[tree].E and self.trees[tree].E[current] == x_init:
                break
        path.append(x_init)
        path.reverse()
        return path

    def check_solution(self):
        # probabilistically check if solution found
        # if self.prc and random.random() < self.prc:
        if self.prc and random.random() < 2:
            # print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            # check if path is collision-free

            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None

    def integral_from_x(self, x):
        # dt = 0.2
        dt = 0.07
        step = self.DS_step
        eps = 0.01  # 0.57295779 degree
        x_next = x  # use the newest point to integral
        # x_next = np.array(self.get_nearest(0, self.x_goal))  # use the closed point to integral
        b_reached = False
        x_DS = []
        # self.trees.append(copy.copy(self.trees[0]))  # create another tree to store the DS traj
        while step > 0 and not b_reached:
            x_DS.append(x_next)
            dq = self.vel(x_next, self.X.x_obj)
            x_last = x_next
            x_next = x_next + dq * dt

            # self.add_vertex(1, tuple(x_next))
            # self.add_edge(1, tuple(x_next), tuple(x_last))
            b_reached = np.linalg.norm(x_next - np.array(self.x_goal)) < eps
            if b_reached:
                x_DS.append(np.array(self.x_goal))
                break
            step = step - 1
        if b_reached:
            print('reach the goal', np.linalg.norm(x_next - np.array(self.x_goal)), str(self.samples_taken), "samples")
            # add the DS traj to the tree
            # self.trees[0] = self.trees[1]
            # del self.trees[1]
            for i in range(1, len(x_DS)):
                self.add_vertex(0, tuple(x_DS[i]))
                self.add_edge(0, tuple(x_DS[i]), tuple(x_DS[i - 1]))

            return True
        else:
            print('do not reach the goal', np.linalg.norm(x_next - np.array(self.x_goal)), str(self.samples_taken),
                  "samples")
            return False

            # print('vel', dq)

    def solution_DS_check_all(self, x):
        x_ = np.vstack(list(x))
        # if self.prc and random.random() < self.prc:
        if self.prc and random.random() < 100:
            #     dt = 0.2
            dt = 0.1
            # dt = 0.2
            step = self.DS_step
            eps = 0.04  # 0.57295779 degree
            x_DS = np.zeros([x_.shape[0], x_.shape[1], step + 1])# (n,2,steps)
            x_integral = np.copy(x_)
            x_DS[:, :, 0] = np.copy(x_)
            for i in range(1, step + 1):
                dq = self.vel(x_integral, self.X.x_obj)
                if self.dim == 8:
                    dq[4:12] = np.zeros(8)
                if self.dim == 4:
                    dq[4:] = np.zeros(12)
                if self.dim ==2 :
                    dq[0] = 0
                    dq[3:] = np.zeros(13)
                x_integral = x_integral + dq * dt
                x_integral = self.bound_point(x_integral)
                x_DS[:, :, i] = np.copy(x_integral)
                c = np.linalg.norm(x_DS[:, :, i] - np.array(self.x_goal), axis=1) < eps
                if any(c):

                    break
            # c_x = dict(zip(x_DS[:, :, 0], c))
            if any(c):
                # path_check = np.concatenate((x_DS[:, 0, :i + 1].flatten().reshape(-1,1), x_DS[:, 1, :i + 1].flatten().reshape(-1,1)) , axis=1)


                print('find available DS connection')
                tmp = x_DS[c, :, :]
                f_c = [path_cost(self.trees[0].E, self.x_init, tuple(x_c)) for x_c in tmp[:, :, 0]]
                c_i = np.argmin(f_c)
                x_DS_best = tmp[c_i, :, :i+1]  # (2x101)
                x_DS_best = np.concatenate([x_DS_best, np.array(self.x_goal).reshape(-1, 1)], axis=1).T
                pairs = self.get_dis(x_DS_best, gradient=False, x_obj=self.X.x_obj,
                                     )
                # for i in range(1, x_DS_best.shape[1]):
                #     self.add_vertex(0, tuple(x_DS_best[:, i]))
                #     self.add_edge(0, tuple(x_DS_best[:, i]), tuple(x_DS_best[:, i - 1]))
                # path = self.reconstruct_path(0, self.x_init, self.x_goal)
                if pairs > 0:
                    count = -1
                    for num, a in enumerate(c):
                        if a:
                            count+=1
                        if count==c_i:
                            break
                    a1 = list(x)
                    a2 = a1[num]
                    path = self.reconstruct_path(0, self.x_init, a2)

                    if path is not None:

                        return True, path + [tuple(x_DS_best[i, :]) for i in range(0, x_DS_best.shape[0])], len(path), x_DS_best, path
        return False, None

    def solution_DS_check_all_(self, x):
        x_new = x
        x = np.array(x)
        # if self.prc and random.random() < self.prc:
        if self.prc and random.random() < 100:
            #     dt = 0.2
            # dt = 0.2
            dt = 0.1
            # dt = 0.2
            step = self.DS_step
            # eps = 0.04 *2 # 0.57295779 degree
            eps = 0.04 # 0.57295779 degree
            x_DS = np.zeros([step + 1, x.shape[0]])
            x_DS[0, :] = np.copy(x)
            for i in range(1, step + 1):
                dq = self.vel(x, self.X.x_obj)
                if self.dim == 8:
                    dq[4:12] = np.zeros(8)
                if self.dim == 4:
                    dq[4:] = np.zeros(12)
                if self.dim ==2:
                    dq[0] = 0
                    dq[3:] = np.zeros(13)
                x = x + dq * dt
                x = self.bound_point(x)
                x_DS[i, :] = np.copy(x)
                # c = np.linalg.norm(x_DS[:, :, i] - np.array(self.x_goal), axis=1) < eps
                c = np.linalg.norm(x_DS[i, :] - np.array(self.x_goal)) < eps
            # if any(c):
                if c:
                    break
            # c_x = dict(zip(x_DS[:, :, 0], c))
            # c_x = dict(zip(x_DS[:, :, 0], c))
            if c:
                print('find available DS connection')
                # tmp = x_DS[c, 0, :]
                # f_c = [path_cost(self.trees[0].E, self.x_init, tuple(x_c)) for x_c in tmp[:, :, 0]]
                # c_i = np.argmin(f_c)
                x_DS_best = x_DS[:i+1, :]  # (101x2)
                x_DS_best = np.concatenate([x_DS_best, np.array(self.x_goal).reshape(1, -1)], axis=0)
                # self.   may need to check if the last point to the goal is collision-free
                # for i in range(1, x_DS_best.shape[0]):
                #     self.add_vertex(0, tuple(x_DS_best[i, :]))
                #     self.add_edge(0, tuple(x_DS_best[i, :]), tuple(x_DS_best[i - 1, :]))
                # path = self.reconstruct_path(0, self.x_init, self.x_goal)
                pairs = self.get_dis(x_DS_best, gradient=False, x_obj=self.X.x_obj,
                                     )
                if pairs > 0:
                    path = self.reconstruct_path(0, self.x_init, x_new)

                    if path is not None:
                        return True, path + [tuple(x_DS_best[i, :]) for i in range(1, x_DS_best.shape[0])], len(path), x_DS_best
                # else:

        return False, None

    def solution_DS_check(self, x):
        # if self.prc and random.random() < self.prc:
        if self.prc and random.random() < 2:
            b_reached = self.integral_from_x(x)
            path = None
            if b_reached:
                path = self.reconstruct_path(0, self.x_init, self.x_goal)
            if path is not None:
                return True, path

        return False, None

    def bound_point(self, point):
        # if point is out-of-bounds, set to bound
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)

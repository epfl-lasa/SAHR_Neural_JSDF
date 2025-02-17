import numpy as np
import time
import torch
# from rtree import index
# https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Index.nearest

from motion_planning.PRM.D_star_lite import DStar
from motion_planning.PRM.KNN_test import FaissKNeighbors

from motion_planning.NN_model.nn_model_eval import NN_hand_obj
from motion_planning.NN_model.nn_model_eval import NN_SCA_pro

import matplotlib.pyplot as plt



class PRM_tool_fast:
    def __init__(self, start_q, end_q, x_obj, right_hand=False, PRM_para=None, use_cuda=True) -> None:
        pass
        if PRM_para is None:
            PRM_para = [400, 3, 300]
        n = PRM_para[0]
        edge_sample_num = PRM_para[1]
        self.edge_sample_num = edge_sample_num
        self.step_max = PRM_para[2]
        self.start_q = tuple(start_q)
        self.end_q = tuple(end_q)
        self.use_cuda = use_cuda
        
        self.hand = collision_check(x_obj, use_cuda=use_cuda, right_hand=right_hand)
        # generate samples
        dim = 16
        lb= np.copy(self.hand.nn.hand_bound[0, :])
        ub = np.copy(self.hand.nn.hand_bound[1, :])
        lb[:4] = np.zeros(4)
        ub[:4] = np.zeros(4) + 0.01
        lb[12:] = np.array([0.5,0,0,0])
        ub[12:] = np.array([0.5,0,0,0]) + 0.01

        self.samples = np.random.uniform(lb, ub, size=(n, dim))
        self.samples = np.concatenate([np.array(start_q).reshape(1, -1), np.array(end_q).reshape(1, -1), self.samples])

        # knn
        k0 = int(np.ceil(np.e * (1 + 1 / dim))) + 2
        knn = FaissKNeighbors(k=k0 + 1)
        knn.fit(self.samples)
        samples_near = knn.predict(self.samples)[:, 1:, :]  # remove the first one, which is itself
        s1 = list(map(tuple, self.samples))
        s2 = [list(map(tuple, samples_near[i, :, :])) for i in range(samples_near.shape[0])]
        self.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...} nearby nodes

        start_p = np.repeat(self.samples, repeats=k0, axis=0)
        edge_samples_ = np.linspace(start_p, np.vstack(samples_near),
                                edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
        self.edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)
        self.start_end_p = [] # a list of tuple for all edges
        for i in range(len(s2)):
            for j in range(k0):
                self.start_end_p.append(s1[i] + s2[i][j])

        graph_D = graph(dim=dim)
        graph_D.edges = self.edges

        self.hand.q_gpu = torch.Tensor(self.edge_samples).to('cuda:0') if use_cuda else torch.Tensor(self.edge_samples)
        self.d_star = DStar(self.start_q, self.end_q, graph_D, "euclidean")

        self.pairs_last = []
        self.path = None

    def run(self, x_obj: np.ndarray, safety_dis=0):
        self.hand.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj) # update obstacle positions.

        pairs = self.hand.collision_hand_obj_SCA(q=None, sample=self.edge_sample_num, safety_dis=safety_dis)
        self.d_star.graph.E = dict(zip(self.start_end_p, pairs))
        if len(self.pairs_last) != 0:
            diff = np.logical_xor(pairs, self.pairs_last)  # True means the state is changed
            # store the edges that need to update
            edges2update = [self.start_end_p[i] for i in range(len(self.start_end_p)) if diff[i]]
            # print('edge num to be updated', len(edges2update))
            self.d_star.update_cost(edges2update)
        self.pairs_last = pairs

        result = self.d_star.ComputePath()
        path = self.d_star.extract_path(max_nums=self.step_max)
        self.path = path
        if len(path) >= self.step_max or len(path) < 2:
            pass
            print('No feasible path.')
            return None
        else:
            return path
    
    def plot_isoline_PRM(self, q_now, x_obj, joint_index=[9,10], name_suffix=1, q_g_real=None):
        save_path = 'figures/obs_PRM_static/'
        fig, ax = plt.subplots(figsize=(8, 6))
        output = self.hand.get_full_map_dis(q_now, x_obj) - 0.015
        con = plt.contourf(self.hand.x_grid, self.hand.y_grid, output, np.linspace(-0.014, 0.1, 10), cmap='PiYG')

        plt.contour(con, levels=[0], colors=('k',), linestyles=('-',),
                    linewidths=(2,))  # boundary of the obs in C space
        plt.title(label='Isolines and robot trajectory')
        cax = plt.axes([0.95, 0.1, 0.05, 0.8])
        plt.colorbar(con, cax=cax)
        plt.title(label='Dis [m]')

        ax.set_xlabel('Ring finger $q_1$' + ' (rad)')
        ax.set_ylabel('Ring finger $q_2$' + ' (rad)')
        # for k, v in self.d_star.graph.edges.items():
        #     for vi in v:
        #         if self.d_star.graph.E[k + vi]:
        #             ax.plot([k[joint_index[0]], vi[joint_index[0]]], [k[joint_index[1]], vi[joint_index[1]]], color='gray')
        ax.scatter(self.samples[:, 0], self.samples[:, 1], c='k', zorder=100)  # all nodes of PRM
        ax.scatter(self.start_q[joint_index[0]], self.start_q[joint_index[1]], c='b', zorder=100)
        ax.scatter(self.end_q[joint_index[0]], self.end_q[joint_index[1]], c='r', zorder=100)
        ax.scatter(q_now[joint_index[0]], q_now[joint_index[1]], c='k', zorder=100)
        if q_g_real is not None:
            ax.scatter(q_g_real[0], q_g_real[1], c='pink', zorder=100, marker=(5, 1),s =40)
        # ax.scatter(s_robot[0], s_robot[1], c='k', zorder=100)
        # plot feasible path
        if self.path is not None:
            for i in range(len(self.path) - 1):
                ax.plot([self.path[i][joint_index[0]], self.path[i + 1][joint_index[0]]], [self.path[i][joint_index[1]], self.path[i + 1][joint_index[1]]], color='b')
                ax.scatter(self.path[i][joint_index[0]], self.path[i][joint_index[1]], c='r', zorder=100)

        fig.savefig(save_path + 'regrasp_obs_PRM_' + str(name_suffix) +  '.png', format='png', bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300)
        plt.close()








class PRM_tool:
    def __init__(self, start_q, end_q, x_obj, g=None, use_cuda=True, right_hand=False, PRM_para=None, dofs=None):
        if PRM_para is None:
            PRM_para = [400, 3, 300]
        n = PRM_para[0]
        k0 = int(np.ceil(np.e * (1 + 1 / len(start_q)))) + 7
        edge_sample_num = PRM_para[1]
        self.edge_sample_num = edge_sample_num
        self.step_max = PRM_para[2]
        self.start_q = tuple(start_q)
        self.end_q = tuple(end_q)

        self.use_cuda = use_cuda
        self.x_obj = x_obj
        self.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        if g is None:
            g = [0, 1, 2, 3, 4]  # use the full hand

        if end_q is None:
            end_q = np.zeros(16)
            end_q[12] = 0.6
        # collision_checke_cuda=use_cuda, right_hand=right_hand)
        self.hand = collision_check(x_obj, use_cuda=use_cuda, right_hand=right_hand, g=g)
        # generate samples
        # assert len(g) == 5
        dim = len(start_q)
        # samples = np.random.uniform(self.hand.nn.hand_bound[0, :], self.hand.nn.hand_bound[1, :], size=(n, dim))

        # generate graph
        graph_D = graph(dim=dim)
        samples = np.random.uniform(self.hand.nn.hand_bound[0, :], self.hand.nn.hand_bound[1, :], size=(n, 16))  # (n, dim)
        samples = samples[:, dofs]
        breakpoint()
        samples = np.concatenate(
            [np.array(start_q).reshape(1, -1), np.array(end_q).reshape(1, -1), samples])  # (n+2, dim)
        knn = FaissKNeighbors(k= k0 + 1)  # knn
        samples = samples.copy(order='C')
        knn.fit(samples)
        samples_near = knn.predict(samples)[:, 1:, :]  # remove the first one, which is itself
        s1 = list(map(tuple, samples))
        s2 = [list(map(tuple, samples_near[i, :, :])) for i in range(samples_near.shape[0])]

        graph_D.edges = dict(zip(s1, s2))  # {v1: [a1,a2,a3], v2; [...],...}
        start_p = np.repeat(samples, repeats=k0, axis=0)
        edge_samples_ = np.linspace(start_p, np.vstack(samples_near),
                                    edge_sample_num, axis=1)  # (n*k, edge_sample_num, dim)
        edge_samples = np.vstack(edge_samples_)  # (n*k*edge_sample_num, dim)

        self.start_end_p = []  # a list of tuple for all edges
        for i in range(len(s2)):
            for j in range(k0):
                self.start_end_p.append(s1[i] + s2[i][j])
        print('edge_samples shape:', edge_samples.shape)
        self.edge_samples = edge_samples

        # print('K-NN time:', time.time() - t0)

        self.hand.q_gpu = torch.Tensor(edge_samples).to('cuda:0') if use_cuda else torch.Tensor(edge_samples)

        self.d_star = DStar(self.start_q, self.end_q, graph_D, "euclidean")

        self.pairs_last = []

    def run_moving_obs(self, x_obj: np.ndarray, safety_dis=0):
        # input the obstacle in base frame of hand
        # self.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        self.hand.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        # update the collision state for all edges
        # t0 = time.time()
        pairs = self.hand.collision_hand_obj_SCA(q=None, sample=self.edge_sample_num, safety_dis=safety_dis, no_SCA=True)
        # t1 = time.time() - t0
        # print('collision check time cost:', time.time()-t0)q_gpu
        # print('edge num to be updated', len(edges2update))
        # self.d_star.update_cost(edges2update)
        pairs_last = pairs
        self.d_star.graph.E = dict(zip(self.start_end_p, pairs))
        result = self.d_star.ComputePath()
        path = self.d_star.extract_path(max_nums=self.step_max)
        if len(path) >= self.step_max or len(path) < 2:
            pass
            print('No feasible path.')
            return None
        else:
            return path


class collision_check:
    def __init__(self, x_obj, g=None, use_cuda=False, right_hand=True, visualize=[2]):
        if g is None:
            g = [0, 1, 2, 3, 4]
        self.nn = NN_hand_obj(g=g, path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=use_cuda,
                              right=right_hand)
        if right_hand:
            self.nn_SCA = NN_SCA_pro(path_check_point='../NN_model/models/model_new_01_N5.pt', use_cuda=use_cuda,
                                     right_hand=right_hand)
        else:
            self.nn_SCA = NN_SCA_pro(path_check_point='../NN_model/models/model_new_01_N5_left.pt', use_cuda=use_cuda,
                                     right_hand=right_hand)
        self.use_cuda = use_cuda
        self.g = g
        self.x_obj = x_obj
        self.x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        self.q_gpu = []

        nums = 300
        self.nums = nums
        lb = self.nn.hand_bound[0, (visualize[0] - 1) * 4 + 1:visualize[0] * 4 - 1]
        ub = self.nn.hand_bound[1, (visualize[0] - 1) * 4 + 1:visualize[0] * 4 - 1]
        x_ = np.linspace(lb[0], ub[0], nums)
        y_ = np.linspace(lb[1], ub[1], nums)
        x_grid, y_grid = np.meshgrid(x_, y_, )
        self.x1 = x_grid.flatten().reshape(-1, 1)
        self.y1 = y_grid.flatten().reshape(-1, 1)
        # q_ = np.concatenate([np.zeros([nums ** 2, 5]), self.x1, self.y1, np.zeros([nums ** 2, 9])], axis=1)
        q_ = np.concatenate([np.zeros([nums ** 2, 1]), self.x1, self.y1, np.zeros([nums ** 2, 1])], axis=1)
        self.q_grid_gpu = torch.Tensor(q_).to('cuda:0') if use_cuda else torch.Tensor(q_)
        self.q_gpu=[]
        self.x_grid = x_grid
        self.y_grid = y_grid

    def get_full_map_dis(self, q_now=None, x_obj_=None):
        if q_now is None:
            q_now = np.zeros(16)
            q_now[12] = 0
        q_all = np.concatenate([np.repeat(q_now[:9].reshape(1, -1), self.nums**2, axis=0),self.x1, self.y1, np.repeat(q_now[11:].reshape(1,-1), self.nums**2, axis=0)], axis=1)
        q_all_gpu = torch.Tensor(q_all).to('cuda:0') if self.use_cuda else torch.Tensor(q_all)
        q_all_gpu = self.q_grid_gpu
        # input obstacles
        if x_obj_ is None:
            x_obj_ = self.x_obj_gpu
        if not isinstance(x_obj_, torch.Tensor):
            x_obj_ = torch.Tensor(x_obj_).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj_)
        dis = self.nn.eval_multiple(q_all_gpu, x_obj_, real_dis=False, only_min_dis=True, gradient=False)
        return dis.reshape(self.nums, self.nums)

    def get_dis(self, q=None, gradient=False, x_obj=None, sample=None, real_dis=False, safety_dis=0, dx=False):
        if safety_dis != 0:
            real_dis = True
        if x_obj is not None:
            # x_obj_gpu = x_obj
            x_obj_gpu = torch.Tensor(x_obj).to('cuda:0') if self.use_cuda else torch.Tensor(x_obj)
        else:
            x_obj_gpu = self.x_obj_gpu

        # global x_obj_gpu
        if q is None:
            # this is for PRM samples
            output = self.nn.eval_multiple(self.q_gpu, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                           gradient=False)
            if sample is None:
                return output
            else:
                output_bool = output > safety_dis
                output_bool = output_bool.reshape(-1, sample)
                pairs = np.all(output_bool, axis=1)
                return pairs
        else:
            if isinstance(q, tuple):
                q = np.array(q)
            if len(q.shape) == 1:
                q = q.reshape(1, -1)

            if q.shape == (1, 2) or q.shape == (1, 4):
                if q.shape == (1, 2):
                    q = np.array([0, q[0, 0], q[0, 1], 0])
                if gradient:
                    # output, grad = self.nn.eval(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                    #                             gradient=gradient)  # single q, multiple x
                    # return output[self.g[0]], grad[self.g[0]][1:3]
                    output, grad = self.nn.eval_multiple(torch.Tensor(q.reshape(1, -1)), x_obj_gpu, real_dis=real_dis,
                                                         only_min_dis=True,
                                                         gradient=gradient)  # single q, multiple x
                    if q.shape == (1, 2):
                        return output[0], grad[0, 1:3]
                    else:
                        return output[0], grad[0, :]
                else:
                    output = self.nn.eval(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True, gradient=gradient)
                    return output[self.g[0]]
            elif q.shape[0] > 1:
                if q.shape[1] == 16:  # (n, 16)
                    pass
                    q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                    if gradient:
                        output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=False, only_min_dis=True,
                                                             gradient=gradient, dx=dx)  # multiple q, single x
                        return output, grad
                        # if dx:
                        #     return output, grad
                        # else:
                        #     return output, grad[:, :16]
                    else:
                        output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                       gradient=False)  # multiple q,  multiple x
                        if sample is None:
                            # return np.min(output)
                            return output
                        else:
                            output_bool = output > safety_dis
                            output_bool = output_bool.reshape(-1, sample)
                            pairs = np.all(output_bool, axis=1)  # this way is 4 times faster than below
                            # pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
                            #          range(int(n / sample))]
                            return pairs  # bool type for collision

                elif q.shape[1] == 2:  # nx2
                    n = q.shape[0]
                    q = np.concatenate([np.zeros([n, 1]), q, np.zeros([n, 1])], axis=1)
                    q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                    if gradient:
                        output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                             gradient=gradient, dx=dx)  # multiple q, single x
                        if dx:
                            return output, grad
                        else:
                            return output, grad[:, 1:3]
                    else:
                        output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                       gradient=False)  # multiple q,  multiple x
                        if sample is None:
                            # return np.min(output)
                            return output
                        else:
                            output_bool = output > safety_dis
                            output_bool = output_bool.reshape(-1, sample)
                            pairs = np.all(output_bool, axis=1)  # this way is 4 times faster than below
                            # pairs = [all(output[j * sample: (j + 1) * sample] > 0) for j in
                            #          range(int(n / sample))]
                            return pairs  # bool type for collision
                else:
                    raise NotImplementedError
            elif q.shape == (1, 16):
                q = torch.Tensor(q).to('cuda:0') if self.use_cuda else torch.Tensor(q)
                if gradient:
                    output, grad = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                gradient=gradient)  # single q, multiple x
                    return output, grad
                else:
                    output = self.nn.eval_multiple(q, x_obj_gpu, real_dis=real_dis, only_min_dis=True,
                                                gradient=gradient)
                    return output

            else:
                raise ValueError('q has a wrong shape', q.shape)

    def obstacle_free(self, q):
        """
        Check if a location resides inside of an obstacle
        :param q: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        # return self.obs.count(x) == 0
        return self.get_dis(q, x_obj=self.x_obj) > 0

    def SCA_eval(self, q=None, gradient=False, sample=None, safety_dis=0):
        if q is None:
            q = self.q_gpu
        real_dis = False
        if safety_dis != 0:
            real_dis = True
        if gradient:
            output, grad = self.nn_SCA.eval(q, gradient=gradient, real_dis=real_dis)
        else:
            output = self.nn_SCA.eval(q, gradient=gradient, real_dis=real_dis)

        if sample is None:
            # return np.min(output)
            if gradient:
                return output, grad
            else:
                return output
        else:
            output_bool = output > safety_dis
            output_bool = output_bool.reshape(-1, sample)
            pairs = np.all(output_bool, axis=1)
            return pairs

    def collision_hand_obj_SCA(self, q=None, sample=None, safety_dis=0, no_SCA=False):
        pair_1 = self.get_dis(q=q, sample=sample, safety_dis=safety_dis)
        if no_SCA:
            return pair_1
        pair_2 = self.SCA_eval(q=q, sample=sample, safety_dis=safety_dis)

        return np.logical_and(pair_1, pair_2)



class graph:
    def __init__(self, dim=2):
        # p = index.Property()
        # p.dimension = dim
        # self.V = index.Index(interleaved=True, properties=p)  # vertices in a rtree
        self.dim = dim
        self.V_count = 0

        self.E = {}  # edges  {v : [v1, v2, v2,..., vn], v' :  } with collision-free edges
        self.edges = {}  # edges  {v : [v1, v2, v2,..., vn], v' :  } with collision-free edges
        self.V_dis_grad = {}  # store the distance and gradients for all vertices

    # def nearby(self, q, k):
    #     #  return the near vertices by knn
    #     # k(n) := k_{PRM} * log(n) , k_{PRM} > e(1 + 1/dim)
    #     return self.V.nearest(q, num_results=k, objects='raw')

    # def add_vertex(self, v):
    #     self.V.insert(0, v + v, v)
    #     self.V_count += 1


class plot_figure:
    def __init__(self):
        pass

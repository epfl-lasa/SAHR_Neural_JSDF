import numpy as np
import torch
from motion_planning.NN_model.nn_model_eval import NN_hand_obj


class point_obstacle:
    def __init__(self, g=None, use_cuda=False):
        if g is None:
            g = [0, 1, 2, 3, 4]
        self.g = g
        self.hand_obj_nn = NN_hand_obj(g=self.g, x_obj_normalized=False,
                                       path_prefix_suffix=['../NN_model/models/single_', '01.pt'], use_cuda=use_cuda)
        if 0 in g:
            self.dim = (len(g) - 1) * 4
        else:
            self.dim = len(g) * 4
        self.dis = np.zeros(self.dim//4)
        self.grad = np.zeros(self.dim)

    def dis_and_grad(self, q, x, real_dis=False, whole_hand=False):
        # only_min_dis, only get the min distance between the obj and meshes of one finger
        dis, grad = self.hand_obj_nn.eval(q, x, real_dis=real_dis, gradient=True, only_min_dis=True)
        dis_ = []
        grad_ = []
        for g in self.g:
            if g != 0:
                dis_.append(dis[g])
                grad_.append(grad[g][:4])  # don't need the gradient wrt object
        # self.dis =
        # min_dis = np.argmin(dis_)
        if whole_hand:
            dis_min = np.min(dis_)
            indices = np.argmin(dis_)
            for i, grad in enumerate(grad_):
                if i != indices:
                    grad_[i] = np.zeros(4)
            return dis_min, np.hstack(grad_)
        else:
            return dis_, grad_

    # def dis_multiple(self, q, x):








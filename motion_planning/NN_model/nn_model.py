import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Net_1s(nn.Module):
    def __init__(self, dim_in, dim_out, layer_width=None):
        super(Net_1s, self).__init__()
        # input q,x,
        # output 22 dis
        # build NN
        if layer_width is None:
            layer_width = [250, 130, 50, 25]
        # self.layers = []
        # layer_width = [dim_in] + layer_width + [dim_out]
        # for i in range(len(layer_width) - 1):
        #     self.layers.append(nn.Linear(layer_width[i], layer_width[i+1]))
        # self.layers_num = len(self.layers)
        self.l1 = nn.Linear(dim_in, layer_width[0])
        self.l2 = nn.Linear(layer_width[0], layer_width[1])
        self.l3 = nn.Linear(layer_width[1], layer_width[2])
        self.l4 = nn.Linear(layer_width[2], layer_width[3])
        self.l5 = nn.Linear(layer_width[3], dim_out)
        # self.act = nn.Tanh()
        self.act_relu = nn.ReLU()
        # self.act_leaky_relu = nn.LeakyReLU()
        # self.lsm = nn.LogSoftmax()
        # self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        # input q,x,
        # output 22 dis
        # for i in range(self.layers_num):
        #     x = self.layers[i](x)
        #     if i < self.layers_num - 1:
        #         x = self.act_relu(x)
        x = self.l1(x)
        x = self.act_relu(x)
        x = self.l2(x)
        x = self.act_relu(x)
        x = self.l3(x)
        x = self.act_relu(x)
        x = self.l4(x)
        x = self.act_relu(x)
        x = self.l5(x)
        return x


class Net_5(nn.Module):
    def __init__(self, dim_in, dim_out, layer_width=None, act=1):
        super(Net_5, self).__init__()

        # build NN
        if layer_width is None:
            layer_width = [250, 130, 50, 25]
        # self.layers = []
        # layer_width = [dim_in] + layer_width + [dim_out]
        # for i in range(len(layer_width) - 1):
        #     self.layers.append(nn.Linear(layer_width[i], layer_width[i+1]))
        # self.layers_num = len(self.layers)
        self.l1 = nn.Linear(dim_in, layer_width[0])
        self.l2 = nn.Linear(layer_width[0], layer_width[1])
        self.l3 = nn.Linear(layer_width[1], layer_width[2])
        self.l4 = nn.Linear(layer_width[2], layer_width[3])
        self.l5 = nn.Linear(layer_width[3], dim_out)
        # self.act = nn.Tanh()
        if act == 1:
            self.act_relu = nn.ReLU()
        else:
            self.act_relu = nn.LeakyReLU()
        # self.act_leaky_relu = nn.LeakyReLU()
        # self.lsm = nn.LogSoftmax()
        # self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        # for i in range(self.layers_num):
        #     x = self.layers[i](x)
        #     if i < self.layers_num - 1:
        #         x = self.act_relu(x)
        x = self.l1(x)
        x = self.act_relu(x)
        x = self.l2(x)
        x = self.act_relu(x)
        x = self.l3(x)
        x = self.act_relu(x)
        x = self.l4(x)
        x = self.act_relu(x)
        x = self.l5(x)
        return x

    # def train(self):
    #     pass

    def save_model(self):
        pass

    def load_model_parameters(self):
        pass


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, layer_nums=None):
        super(Net, self).__init__()

        # build NN
        if layer_nums is None:
            layer_nums = [250, 130, 50]
        self.l1 = nn.Linear(dim_in, layer_nums[0])
        self.l2 = nn.Linear(layer_nums[0], layer_nums[1])
        self.l3 = nn.Linear(layer_nums[1], layer_nums[2])
        self.l4 = nn.Linear(layer_nums[2], dim_out)
        # self.act = nn.Tanh()
        self.act_relu = nn.ReLU()
        # self.act_leaky_relu = nn.LeakyReLU()
        # self.lsm = nn.LogSoftmax()
        # self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.l1(x)
        x = self.act_relu(x)
        x = self.l2(x)
        x = self.act_relu(x)
        x = self.l3(x)
        x = self.act_relu(x)
        x = self.l4(x)
        # x = self.lsm(x)
        # x = self.act(x)
        # x = self.act_relu(x)
        return x


def loss_SCA_neg(output, target, loss_weight, threshold=0.5):
    cond1 = torch.logical_and(output <= 0, target <= 0)
    loss_0 = (output - target) ** 2
    cond2 = torch.logical_and(output > threshold, target > threshold)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss_0 * loss_weight[0],
                       loss_0)
    cond3 = torch.logical_and(output > 0, target <= 0)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss * loss_weight[1],
                       loss)

    cond4 = torch.logical_and(output < 0, target > 0)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss * loss_weight[2],
                       loss)
    return torch.sum(torch.mean(loss))


def loss_SCA(output, target, threshold=0.5):
    """
    loss function for self-collision avoidance
     threshold, a positive float,

    """
    cond1 = torch.logical_and(output <= 0, target <= 0)
    cond2 = torch.logical_and(output > threshold, target > threshold)

    loss = torch.where(torch.logical_or(cond1, cond2),
                       torch.zeros(1).cuda(),
                       (output - target) ** 2)

    loss_2 = torch.mul(loss, 5)
    cond3 = torch.logical_and(output <= 0, target > 0)
    loss_new = torch.where(cond3,
                           loss_2,
                           loss)

    return torch.sum(torch.mean(loss_new))


def hand_obj_dis(output, target, c, loss_weight, threshold=0.5):
    # cond1 = torch.logical_and(output <= 0, target <= 0)
    # cond2 = torch.logical_and(output > threshold, target > threshold)
    #
    # loss = torch.where(torch.logical_or(cond1, cond2),
    #                    torch.zeros(1).cuda(),
    #                    (output - target) ** 2)
    #
    # loss_2 = torch.mul(loss, 10)
    # cond3 = torch.logical_and(output <= 0, target > 0)
    # cond4 = torch.logical_and(output > 0, target <= 0)
    # loss_new = torch.where(torch.logical_or(cond3, cond4),
    #                        loss_2,
    #                        loss)
    ##### version with loss_weight
    cond1 = torch.logical_and(output <= 0, target <= 0)
    loss_0 = (output - target) ** 2
    cond2 = torch.logical_and(output > threshold, target > threshold)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss_0 * loss_weight[0],
                       loss_0)
    cond3 = torch.logical_and(output > 0, target <= 0)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss * loss_weight[1],
                       loss)

    cond4 = torch.logical_and(output < 0, target > 0)
    loss = torch.where(torch.logical_or(cond1, cond2),
                       loss * loss_weight[2],
                       loss)
    # cond3 = torch.logical_and(output >= 0, target > 0)
    if c != []:
        loss = torch.mul(loss, c)

    return torch.sum(torch.mean(loss))


def loss_classification(output, target):
    value_mul = torch.mul(output, target)
    # cond1 = torch.logical_and(output <= 0, target > 0)
    # cond2 = torch.logical_and(output > 0, target <=0)
    # cond = torch.logical_or(cond1, cond2)
    return torch.sum(value_mul < 0) / output.shape[0]


def hand_obj_loss(output, target, alpha=0.1, threshold=0.5):
    loss_1 = hand_obj_dis(output, target, threshold=threshold)
    loss_2 = loss_classification(output, target)
    return loss_1 + loss_2 * alpha


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def validate(outputs, y_test_cpu, dis_scale, threshold=0.5):
    outputs = recover_dis(outputs, dis_scale)
    y_real = recover_dis(y_test_cpu, dis_scale)

    outputs[outputs <= 0] = 0
    y_real[y_real <= 0] = 0
    error = (y_real - outputs)
    print('     1. dis error', np.sqrt(np.mean(error * error, axis=0)))

    dis_bool = y_real <= 0
    dis_bool_p = outputs <= 0
    c = np.equal(dis_bool, dis_bool_p)
    print('     2. collision accuracy for all dis', np.sum(c) / c.shape[0] / c.shape[1])

    # only compare the minimal distance
    dis_bool = np.min(y_real, axis=1) <= 0
    dis_bool_p = np.min(outputs, axis=1) <= 0
    c = np.equal(dis_bool, dis_bool_p)
    print('     3. collision accuracy for min dis', np.sum(c) / len(c))

    true_neg = dis_bool_p[dis_bool]

    dis_bool = np.min(y_real, axis=1) > 0
    dis_bool_p = np.min(outputs, axis=1) > 0
    true_pos = dis_bool_p[dis_bool]

    FN = 1 - np.sum(true_pos) / len(true_pos)
    FP = 1 - np.sum(true_neg) / len(true_neg)
    print('False positive:', FP)
    print('False negative:', FN)


def recover_dis(output, dis_scale):
    output = np.copy(output)
    for i in range(dis_scale.shape[1]):
        neg = output[:, i] < 0
        output[neg, i] = - output[neg, i] * dis_scale[0, i]
        pos = output[:, i] > 0
        output[pos, i] = output[pos, i] * dis_scale[1, i]
    return output


def validate_SCA_neg(outputs, y_test_cpu, dis_scale, record=False):
    outputs = recover_dis(outputs, dis_scale)
    y_real = recover_dis(y_test_cpu, dis_scale)
    error = y_real - outputs
    if record:
        RMSE = np.sqrt(np.mean(error * error))
    else:
        RMSE = np.sqrt(np.mean(error * error, axis=0))
    print('     1. dis error', RMSE)

    dis_bool = y_real <= 0
    dis_bool_p = outputs <= 0
    c1 = np.equal(dis_bool, dis_bool_p)
    c1 = np.sum(c1) / c1.shape[0] / c1.shape[1]
    print('     2. collision accuracy for all dis', c1)

    # only compare the minimal distance
    dis_bool = np.min(y_real, axis=1) <= 0
    dis_bool_p = np.min(outputs, axis=1) <= 0
    c = np.equal(dis_bool, dis_bool_p)
    c2 = np.sum(c) / len(c)
    print('     3. collision accuracy for min dis', c2)

    true_neg = dis_bool_p[dis_bool]

    dis_bool = np.min(y_real, axis=1) > 0
    dis_bool_p = np.min(outputs, axis=1) > 0
    true_pos = dis_bool_p[dis_bool]

    FN = 1 - np.sum(true_pos) / len(true_pos)
    FP = 1 - np.sum(true_neg) / len(true_neg)
    print('False positive:', FP)
    print('False negative:', FN)

    if record:
        acc = np.array([c1, c2, FP, FN])
        return RMSE, acc


def load_dataset_SCA(name, path='../dataset/', add_sin_cos=3, normalization=0, dec=1):
    data2 = np.load(path + name)  # # 16 joints, 10 min_dis
    nums = int(data2.shape[0] * dec)
    data2 = data2[:nums, :]
    print('dataset shape:', data2.shape, 'load data from', path + name)
    print('nonzero dis:', np.count_nonzero(np.min(data2[:, 16:], axis=1) > 0) / nums)
    # add sin cos
    hand_joint_bounds = np.loadtxt('models/hand_joint_bound.txt')
    lb = hand_joint_bounds[0, :]
    ub = hand_joint_bounds[1, :]
    if add_sin_cos == 3:  # [q, sin(q), cos(q)]
        num_s = 16 * 3
        data_q = np.concatenate([data2[:, :16], np.sin(data2[:, :16]), np.cos(data2[:, :16])], axis=1)
    elif add_sin_cos == 2:  # [sin(q), cos(q)]
        num_s = 16 * 2
        data_q = np.concatenate([np.sin(data2[:, :16]), np.cos(data2[:, :16])], axis=1)
    elif add_sin_cos == 1:  # [q] only q
        num_s = 16 * 1
        data_q = data2[:, :16]
    if add_sin_cos == 3 or add_sin_cos == 1:
        for i in range(16):  # joint angles to [-1,1]
            data_q[:, i] = (data_q[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1
    # build q_all
    # normalize dis

    data_dis = data2[:, 16:]
    if normalization == 0:
        pass
    elif normalization == 1:
        dis_min = np.min(data2[:, 16:], axis=0)
        dis_max = np.max(data2[:, 16:], axis=0)
        dis_scale = np.vstack([dis_min, dis_max])
    else:
        dis_scale = np.array([[-0.007, -0.007, -0.007, -0.015, -0.02, -0.01, -0.01, -0.02, -0.02, -0.02],
                              [0.0646, 0.0646, 0.0646, 0.0199, 0.01668, 0.0602, 0.04187, 0.01668, 0.043, 0.057678]])

    if normalization:
        for i in range(10):
            neg = data_dis[:, i] < 0
            data_dis[neg, i] = - data_dis[neg, i] / dis_scale[0, i]
            pos = data_dis[:, i] > 0
            data_dis[pos, i] = data_dis[pos, i] / dis_scale[1, i]

    data = np.concatenate([data_q, data_dis], axis=1)

    return data, num_s, dis_scale


def load_dataset(name, path='../dataset/', add_sin_cos=3, keep_all_dis=True, group=0):
    data2 = np.load(path + name)  # # 16 joints, 3 obj pose, 15 min_dis
    nums = data2.shape[0]
    if name[:3] == 'obj':
        obj = True
        # nums = int(nums * 0.1)
        print('load dataset', name)
    else:
        obj = False
        # nums = 1000000

    data2 = data2[:nums, :]
    print('dataset shape:', data2.shape)

    dis_with_obj_5 = list(np.array([5, 9, 12, 14, 15]) + 18)
    dis_with_obj = [16, 17, 18] + dis_with_obj_5
    # dis_with_obj
    dis_only_hand = []
    for i in range(34):
        if i not in dis_with_obj:
            dis_only_hand.append(i)
    # print(dis_only_hand)
    if obj:
        num_in = 19
        num_all = data2.shape[1]
        obj_bounds = np.vstack([np.min(data2[:, 16:19], axis=0), np.max(data2[:, 16:19], axis=0)])
        dis = data2[:, 19:]
        dis[dis < 1e-5] = 0
        data2[:, 19:] = dis
        data1 = data2  # 16 joints, 3 obj position, 5 dis between obj and hand

    else:
        data1 = data2[:, dis_only_hand]  # 16 joints, 10 min_dis
        num_in = 16
        num_all = data1.shape[1]
    print('nonzero dis:', np.count_nonzero(np.min(data1[:, num_in:], axis=1)) / nums)

    # max_dis = [np.max(data1[:, i]) for i in range(num_in, num_all)]
    max_dis = np.max(data1[:, num_in:], axis=0)

    # normalize data to [-1,1]

    hand_joint_bounds = np.loadtxt('models/hand_joint_bound.txt')
    lb = hand_joint_bounds[0, :]
    ub = hand_joint_bounds[1, :]
    data = np.copy(data1)

    data_dis = data1[:, num_in:] / max_dis  # normalize for each dis
    data_dis[data_dis == 0] = -1
    data[:, num_in:] = data_dis

    if add_sin_cos == 3:  # [q, sin(q), cos(q)]
        num_s = 16 * 3
        data_q = np.concatenate([data[:, :16], np.sin(data[:, :16]), np.cos(data[:, :16])], axis=1)
    elif add_sin_cos == 2:  # [sin(q), cos(q)]
        num_s = 16 * 2
        data_q = np.concatenate([np.sin(data[:, :16]), np.cos(data[:, :16])], axis=1)
    elif add_sin_cos == 1:  # [q] only q
        num_s = 16 * 1
        data_q = data[:, :16]

    if keep_all_dis:
        data_dis = data[:, num_in:]
    else:
        data_dis = np.min(data[:, num_in:], axis=1).reshape(-1, 1)

    if add_sin_cos == 3 or add_sin_cos == 1:
        for i in range(16):  # joint angles to [-1,1]
            data_q[:, i] = (data_q[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1

    #
    # if add_sin_cos:
    #     num_s = 16 * 3
    #     if keep_all_dis:
    #         data = np.concatenate([data[:, :16], np.sin(data[:, :16]), np.cos(data[:, :16]), data[:, num_in:]], axis=1)
    #     else:
    #         min_dis = np.min(data1[:, num_in:], axis=1).reshape(-1, 1)
    #         data = np.concatenate([data[:, :16], np.sin(data[:, :16]), np.cos(data[:, :16]), min_dis], axis=1)
    # else:
    #     num_s = 16
    #     if keep_all_dis is False:
    #         min_dis = np.min(data1[:, num_in:], axis=1).reshape(-1, 1)
    #         data = np.concatenate([data[:, :num_in], min_dis], axis=1)

    dis_path = 'models/max_dis.txt'
    if obj:
        num_s += 3
        dis_path = dis_path[:-4] + '_obj.txt'
        np.savetxt('models/obj_bound.txt', obj_bounds, delimiter=' ')
        obj_lb = obj_bounds[0, :]
        obj_ub = obj_bounds[1, :]
        for i in range(3):  # obj position to [-1, 1]
            data[:, i + 16] = (data[:, i + 16] - obj_lb[i]) / (obj_ub[i] - obj_lb[i]) * 2 - 1

        if group == 0:  # use all groups
            data = np.concatenate([data_q, data[:, 16:19], data_dis], axis=1)
        elif group == 1:  # only use palm and obj, but palm is fixed, not relevant with q
            num_s = 3
            if data_dis.shape[1] == 5:
                data_dis_group = data_dis[:, 0:1]
                max_dis = max_dis[group - 1: group]
            else:
                mesh2group = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16],
                              [17, 18, 19, 20, 21]]  # base, index, middle, ring, thumb
                data_dis_group = data_dis[:, mesh2group[group - 1]]
                max_dis = max_dis[mesh2group[group - 1]]

            data = np.concatenate([data[:, 16:19], data_dis_group], axis=1)

        else:  # choose groups,  group = 2,3,4,5  [palm, index, middle, ring, thumb]
            if data_dis.shape[0] == 5:
                data_dis_group = data_dis[:, group - 1:group]
                max_dis = max_dis[group - 1: group]
            else:
                assert data_dis.shape[1] == 22
                mesh2group = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16],
                              [17, 18, 19, 20, 21]]  # base, index, middle, ring, thumb
                data_dis_group = data_dis[:, mesh2group[group - 1]]
                max_dis = max_dis[mesh2group[group - 1]]

            finger = np.arange((group - 2) * 4, (group - 1) * 4)
            if add_sin_cos > 1:
                for i in range(1, add_sin_cos):
                    tmp = np.arange((group - 2) * 4 + i * 16, (group - 1) * 4 + i * 16)
                    finger = np.concatenate([finger, tmp])
            print(finger)
            data = np.concatenate([data_q[:, finger], data[:, 16:19], data_dis_group], axis=1)
            num_s = len(finger) + 3

    else:
        data = np.concatenate([data_q, data_dis], axis=1)

    np.savetxt(dis_path, max_dis, delimiter=' ')

    return data, num_s, max_dis

import time
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import sys

sys.path.append("../..")
import kinematics.allegro_hand as allegro

hand = allegro.Robot(use_fingers=[1, 1, 1, 1], path_prefix='../../', all_link_fk=True, meshes=True)
lb = hand.joint_lb
ub = hand.joint_ub

# load data
path = '../dataset/'
dataset_name = 'dataset'
nums = 10000000
suffix = '_15dis_sphere_2'
# dataset_name = 'test100000.txt'  # 59min 100k nums
# data2 = np.loadtxt(path+dataset_name, delimiter=' ')  #
data2 = np.load(path + dataset_name + str(nums) + suffix + '.npy')  # # 16 joints, 3 obj pose, 15 min_dis
print('data2 shape:', data2.shape)
nums = 1000000
data2 = data2[:nums, :]

dis_with_obj = list(np.array([5, 9, 12, 14, 15]) + 18)
dis_with_obj += [16, 17, 18]
# dis_with_obj
dis_only_hand = []
for i in range(34):
    if i not in dis_with_obj:
        dis_only_hand.append(i)
print(dis_only_hand)
data1 = data2[:, dis_only_hand]  # 16 joints, 10 min_dis

print('nonzero:', np.count_nonzero(np.min(data1[:, 16:], axis=1)) / nums)

max_dis = [np.max(data1[:, i]) for i in range(16, 26)]
max_dis = np.asarray(max_dis)
max_dis_ = np.max(max_dis)

# normalize data to [-1,1]

data = np.copy(data1)
for i in range(16):  # joint angles
    data[:, i] = (data2[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1

data_dis = data1[:, 16:] / max_dis  # normalize for each dis of 10
data_dis[data_dis == 0] = -1

data[:, 16:] = data_dis

add_sin_cos = True
keep_all_dis = True

num = int(data.shape[0] * 0.8)
batch_size = 128 * 64 *4
# batch_size = 128 * 1


if add_sin_cos:
    num_s = 16 * 3
    if keep_all_dis:
        data = np.concatenate([data[:, :16], np.sin(data[:, :16]), np.cos(data[:, :16]), data[:, 16:]], axis=1)
    else:
        min_dis = np.min(data1[:, 16:], axis=1).reshape(-1, 1)
        data = np.concatenate([data[:, :16], np.sin(data[:, :16]), np.cos(data[:, :16]), min_dis], axis=1)
else:
    num_s = 16
    if keep_all_dis is False:
        min_dis = np.min(data1[:, 16:], axis=1).reshape(-1, 1)
        data = np.concatenate([data[:, :16], min_dis], axis=1)

x_train = torch.Tensor(data[:num, :num_s])
y_train = torch.Tensor(data[:num, num_s:])
dim_in = x_train.size(1)
dim_out = y_train.size(1)
dataset_train = TensorDataset(x_train, y_train)  # create your datset
loader_train = DataLoader(dataset_train,
                          batch_size=batch_size,
                          num_workers=16,
                          pin_memory=True,
                          prefetch_factor=4,
                          persistent_workers=True)

x_test = torch.Tensor(data[num:, :num_s])
y_test = torch.Tensor(data[num:, num_s:])

print(dim_in, dim_out)
del data2
del data1
del data_dis

use_cuda = True
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

import torch.nn as nn
from nn_model import Net, loss_SCA, data_prefetcher

net = Net(dim_in, dim_out)

weight_p, bias_p = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

import torch.optim as optim

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(),lr=0.01)
# optimizer = optim.Adam(net.parameters(),lr=0.01,weight_decay=0.001)
optimizer = optim.SGD([
    {'params': weight_p, 'weight_decay': 1e-5},
    {'params': bias_p, 'weight_decay': 0}
], lr=15e-2, momentum=0.9)

net.to(device)
x_gpu = x_train.to(device)
y_gpu = y_train.to(device)

error_all = []



t_start = time.time()
i = 0
while 1:
    t0 = time.time()
    prefetcher = data_prefetcher(loader_train)
    x, y = prefetcher.next()
    while x is not None:

        optimizer.zero_grad()  # zero the parameter gradients

        # forward + backward + optimize
        outputs = net(x)
        loss = loss_SCA(outputs, y)  # y[0,0,2,3] out[0,-0.1,3,4]
        # loss= criterion(outputs, y)

        loss.backward()
        optimizer.step()
        # error_all.append(np.sqrt(loss.data.item())*max_dis_)
        x, y = prefetcher.next()

    i += 1
    if (i % 5 == 0):
        print(i, "%0.4f" % (time.time() - t0), np.sqrt(loss.data.item()) * max_dis_)

    if i >100:
        break
print('time cost for each epoch:', (time.time() - t_start) / i)

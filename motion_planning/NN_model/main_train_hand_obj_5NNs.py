import time
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch.nn as nn
from nn_model import Net, Net_5, loss_SCA, hand_obj_loss, hand_obj_dis, loss_classification, load_dataset, validate

nums = 10000000
# batch_size = 1000 * 1000
batch_size = 0
use_cuda = True
# name = 'obj_5_dis_07_half.npy'
# name = 'obj_22_dis_06_half.npy'
name = 'obj_22_dis_08_half.npy'
name = 'obj_22_dis_02_convex.npy'
data_5 = []
max_dis_5 = []
max_dis_ = []
num_s = []
for i in range(1, 6):
    data, num_s_, max_dis = load_dataset(name, add_sin_cos=3, group=i)
    data_5.append(data)
    max_dis_5.append(max_dis)
    max_dis_.append(np.max(max_dis))
    num_s.append(num_s_)

device = torch.device("cuda:0" if use_cuda else "cpu")
print('device:', device, '  data', data.shape, 'num_s', num_s)
np.savetxt('models/max_dis_22.txt', np.hstack(max_dis_5), delimiter=' ')
obj_bound = np.loadtxt('models/obj_bound.txt', delimiter=' ')

num = int(data.shape[0] * 0.8)

# data[:,51] = np.min(data[:,51:], axis=1)
# data = data[:,:52]
x_train = [torch.Tensor(data_5[i][:num, :num_s[i]]) for i in range(5)]
y_train = [torch.Tensor(data_5[i][:num, num_s[i]:]) for i in range(5)]
dim_in = [x_train[i].size(1) for i in range(5)]
dim_out = [y_train[i].size(1) for i in range(5)]

x_test_gpu = [torch.Tensor(data_5[i][num:, :num_s[i]]).to(device) for i in range(5)]
y_test_cpu = [data_5[i][num:, num_s[i]:] for i in range(5)]

if batch_size <= 0:  # put all data into GPU for training if GPU memory is enough
    x_gpu = [x_train[i].to(device) for i in range(5)]
    y_gpu = [y_train[i].to(device) for i in range(5)]
else:  # use DataLoader to send data from CPU to GPU in each batch_id
    dataset_train = [TensorDataset(x_train[i], y_train[i]) for i in range(5)]  # create dataset
    data_loader = [DataLoader(dataset_train[i],
                              batch_size=batch_size,
                              num_workers=16,  # max 16
                              pin_memory=True,
                              prefetch_factor=4,
                              persistent_workers=True,
                              drop_last=True,
                              shuffle=True
                              ) for i in range(5)]

print('Start to build NN:', device, '  dim_in=', dim_in, ',  dim_out=', dim_out)
net = [Net_5(dim_in[i], dim_out[i], layer_width=[250, 130, 50, 20]) for i in range(5)]
# net = Net(dim_in, dim_out)
# net = Net(dim_in, dim_out, layer_nums=[100, 50, 50])



import torch.optim as optim

# initialization for weight
# for m in net.children():
#     if isinstance(m, nn.Linear):
#         nn.init.uniform_(m.weight)

# put all data to GPU memory


# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(),lr=0.01)

para = []
weight_p, bias_p = [], []
for i in range(5):
    para = para + list(net[i].parameters())
    net[i].to(device)
    for name, p in net[i].named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
# tmp = net[].layers.parameters()
# print()

# optimizer = optim.Adam(para, lr=0.1e-3)

optimizer = optim.AdamW([
    {'params': weight_p, 'weight_decay': 1e-6},
    {'params': bias_p, 'weight_decay': 0}],
    lr=0.1e-3)
# optimizer = optim.SGD([
#     {'params': weight_p, 'weight_decay': 1e-5},
#     {'params': bias_p, 'weight_decay': 0}],
#     lr=10e-2, momentum=0.9)

resume = 1
test_accuracy = 1
# path_check_point = 'models/model_obj_02.pt'  # adam modified loss, penalty output<0 and target>0
path_check_point = 'models/model_obj_5NN_07.pt'  # 5 layers

path_save_model = path_check_point[:-3] + '_eval'
epoch = 0
if resume:

    checkpoint = torch.load(path_check_point)
    for i in range(5):
        net[i].load_state_dict(checkpoint['model_state_dict'][i])
        net[i].train()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Load model from', path_check_point)
else:
    print('Create a new model with default initialization')

error_all = []
best_loss = 1
t_start = time.time()
threshold = 0.5

#             y<0   0<y<1/2    y>1/2
#  d <=0      w1      w2         1
#  0<d<1/2    w3       1         1
#  d>1/2      1        1         w1
w1 = 0.01
w2 = 10
w3 = 10
alpha = 1  # for min dis index

# get the min dis index of y_gpu
c = []
if alpha !=1:
    for i in range(5):
        y_gpu_min = torch.argmin(y_gpu[i], dim=1)
        tmp = torch.ones_like(y_gpu[i])
        tmp[:, y_gpu_min] = alpha
        c.append(tmp)
else:
    c = [[],[],[],[],[]]

loss_weight = [w1, w2, w3]  #
while True:

    # loss_weight = np.array([w1, w2, w3]) + np.array([0, 1, 1]) * epoch * 1e-4

    t0 = time.time()

    if batch_size <= 0:
        optimizer.zero_grad()  # zero the parameter gradients
        loss = []
        for i in range(5):
            outputs = net[i](x_gpu[i])
            tmp = hand_obj_dis(outputs, y_gpu[i], c[i], loss_weight, threshold=0.5)
            loss.append(tmp)
        # loss_2 = loss_classification(outputs, y_gpu)
        #  = hand_obj_loss(outputs, y_gpu, threshold=threshold)
        # loss = criterion(outputs, y_gpu)
        loss_all = sum(loss)
        loss_all.backward()
        optimizer.step()
    # else:
    #     for batch_idx, (x, y) in enumerate(data_loader):
    #         optimizer.zero_grad()  # zero the parameter gradients
    #         if use_cuda:
    #             x = x.cuda()
    #             y = y.cuda()
    #         # forward + backward + optimize
    #         outputs = net(x)
    #         # loss = hand_obj_loss(outputs, y, threshold=threshold )
    #         # loss = criterion(outputs, y)
    #         loss = hand_obj_dis(outputs, y_gpu, threshold=0.5)
    #         # loss_2 = loss_classification(outputs, y_gpu)
    #         #  = hand_obj_loss(outputs, y_gpu, threshold=threshold)
    #         # loss = criterion(outputs, y_gpu)
    #         # loss = loss_1 + loss_2 * alpha
    #
    #         loss.backward()
    #         optimizer.step()

    # error_all.append(np.sqrt(loss.data.item())*max_dis_)
    dt = (time.time() - t0)
    if epoch % 100 == 0:
        # dis_loss = np.sqrt(loss.data.item()) * max_dis_
        print(epoch, ' dt:', "%0.4f" % dt, "%0.8f" % np.sqrt(loss_all.data.item()), '  0:', "%0.8f" % (np.sqrt(loss[0].data.item() * max_dis_[0]))
              , '  1:', "%0.8f" % (np.sqrt(loss[1].data.item() * max_dis_[1]))
              , '  2:', "%0.8f" % (np.sqrt(loss[2].data.item() * max_dis_[2]))
              , '  3:', "%0.8f" % (np.sqrt(loss[3].data.item() * max_dis_[3]))
              , '  4:', "%0.8f" % (np.sqrt(loss[4].data.item()) * max_dis_[4]))
        # print(epoch, ' dt:',"%0.4f" % dt, '  mean dis:', "%0.8f" % dis_loss,'classification loss', "%0.8f"%loss_2.data.item())

        if test_accuracy and epoch % 1000 == 0:
            outputs = []
            y_tests = []
            for i in range(5):
                test_num = 5000
                num_all = y_test_cpu[i].shape[0]
                test_num = num_all
                assert test_num <= num_all
                # s1 = np.random.randint(0, num_all - test_num)
                s1 = 0
                x_test = x_test_gpu[i][s1:s1 + test_num, :]
                y_test = y_test_cpu[i][s1:s1 + test_num, :]
                with torch.no_grad():
                    output = net[i](x_test)
                outputs.append(output.cpu().numpy())
                y_tests.append(y_test)
                # print(i)
            validate(np.hstack(outputs), np.hstack(y_tests), np.hstack(max_dis_5), threshold=threshold)
        if loss_all < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': [net[i].state_dict() for i in range(5)],
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_all,
                'max_dis': np.hstack(max_dis_5),
                'obj_bounds': obj_bound,
            }, path_check_point)
            best_loss = loss_all
            # torch.save(net.state_dict(), path_save_model)
    epoch += 1
    if time.time() - t_start > 3600 * 2:
        break
    # if epoch > 100:
    #     break

# print('time cost for each epoch:', (time.time() - t_start) / epoch)

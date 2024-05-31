import time
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch.nn as nn
from nn_model import Net, loss_SCA, hand_obj_loss, hand_obj_dis, loss_classification, load_dataset, validate

# batch_size = 1000 * 1000
batch_size = 0
use_cuda = True
g = 4
dec = 0.1
name = 'obj_single_convex_' + str(g) + '_combined.npy'
data = np.load('../dataset/' + name)
num = int(data.shape[0] * dec)
data = data[:num, :]
dis_num = [2, 5, 5, 5, 5]

if g == 1:
    dis_min = -0.01
    dis_max = 0.1
    dis_scale = np.array([[dis_min] * dis_num[g], [dis_max] * dis_num[g]])
    obj_bound = np.array([[-0.05, 0, 0.05], [0.15, 0.1, 0.25]])
elif g == 4:
    obj_bound = np.array([[-0.05, -0.08, -0.05], [0.15, 0.18, 0.15]])
    dis_scale = np.array([[-0.01, -0.005, -0.005, -0.005, -0.005], [0.15, 0.15, 0.13, 0.13, 0.15]])

elif g == 0:
    obj_bound = np.array([[-0.212, -0.2837, -0.181], [0.212, 0.2837, 0.3026]])
    dis_scale = np.array([[-0.0065, -0.02], [0.3, 0.20]])

normalization = 1
if normalization == 0:
    data_dis = data[:, 19:]
elif normalization == 1:
    data_dis = data[:, 19:]
    for i in range(dis_num[g]):
        neg = data_dis[:, i] < 0
        data_dis[neg, i] = - data_dis[neg, i] / dis_scale[0, i]
        pos = data_dis[:, i] > 0
        data_dis[pos, i] = data_dis[pos, i] / dis_scale[1, i]
else:
    raise NotImplementedError

# data_dis = data[:, 19:]
# dis_max = np.max(data_dis, axis=0)
# data_dis = data_dis / dis_max
# data_dis[data_dis <= 0] = -1
# print('nonzero dis:', np.count_nonzero(data_dis, axis=1) / data.shape[0])

data[:, 19:] = data_dis

# obj position to [-1, 1]
# obj_bound = np.vstack([np.min(data[:, 16:19], axis=0), np.max(data[:, 16:19], axis=0)])
obj_lb = obj_bound[0, :]
obj_ub = obj_bound[1, :]

for i in range(3):  # obj position to [-1, 1]
    data[:, i + 16] = (data[:, i + 16] - obj_lb[i]) / (obj_ub[i] - obj_lb[i]) * 2 - 1
# joint to [-1 1]
hand_joint_bounds = np.loadtxt('models/hand_joint_bound.txt')
# np.savetxt('models/obj_bound.txt', obj_bound, delimiter=' ')

lb = hand_joint_bounds[0, :]
ub = hand_joint_bounds[1, :]
for i in range(16):  # joint angles to [-1,1]
    data[:, i] = (data[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1

if g == 0:  # palm
    index = np.array([16, 17, 18])
elif g == 1:  # index finger
    index = np.array([0, 1, 2, 3, 16, 17, 18])
elif g == 4:  # thumb
    index = np.array([12, 13, 14, 15, 16, 17, 18])
else:
    raise NotImplementedError

device = torch.device("cuda:0" if use_cuda else "cpu")
print('device:', device, '  data', data.shape, ' input index', index)

num = int(data.shape[0] * 0.8)

# data[:,51] = np.min(data[:,51:], axis=1)
# data = data[:,:52]
test_num = data.shape[0] - num
x_train = torch.Tensor(data[:num, index])
y_train = torch.Tensor(data[:num, 19:])
dim_in = x_train.size(1)
dim_out = y_train.size(1)

x_test_gpu = torch.Tensor(data[num:, index]).to(device)
y_test_cpu = data[num:, 19:]

if batch_size <= 0:  # put all data into GPU for training if GPU memory is enough
    x_gpu = x_train.to(device)
    y_gpu = y_train.to(device)
else:  # use DataLoader to send data from CPU to GPU in each batch_id
    dataset_train = TensorDataset(x_train, y_train)  # create dataset
    data_loader = DataLoader(dataset_train,
                             batch_size=batch_size,
                             num_workers=16,  # max 16
                             pin_memory=True,
                             prefetch_factor=4,
                             persistent_workers=True,
                             drop_last=True,
                             shuffle=True
                             )
print('Data shape', data.shape)
print('Start to build NN:', device, '  dim_in=', dim_in, ',  dim_out=', dim_out)
layer_nums = [250, 130, 50]
net = Net(dim_in, dim_out, layer_nums=layer_nums)
net.to(device)

weight_p, bias_p = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

import torch.optim as optim

# initialization for weight
# for m in net.children():
#     if isinstance(m, nn.Linear):
#         nn.init.uniform_(m.weight)

# put all data to GPU memory


# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(),lr=0.01)
# optimizer = optim.Adam(net.parameters(), lr=0.2e-3)
optimizer = optim.AdamW([
    {'params': weight_p, 'weight_decay': 1e-6},
    {'params': bias_p, 'weight_decay': 0}],
    lr=0.1e-3)
# optimizer = optim.SGD([
#     {'params': weight_p, 'weight_decay': 1e-6},
#     {'params': bias_p, 'weight_decay': 0}],
#     lr=50e-2, momentum=0.9)

resume = 0
test_accuracy = 1
# path_check_point = 'models/model_obj_single_01r.pt'  # adam modified loss, penalty output<0 and target>0
model_name = 'single_' + str(g) + '01_grid_search'
path_check_point = 'models/' + model_name + '.pt'  #

# path_save_model = path_check_point[:-3] + '_eval'
epoch = 0
if resume:
    checkpoint = torch.load(path_check_point)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    net.train()
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
w1 = 0.1
w2 = 3
w3 = 2
alpha = 1  # for min dis index

# get the min dis index of y_gpu
y_gpu_min = torch.argmin(y_gpu, dim=1)
c = torch.ones_like(y_gpu)
if alpha != 1:
    c[:, y_gpu_min] = alpha
else:
    c = []

loss_weight = [w1, w2, w3]  #
while True:
    t0 = time.time()

    if batch_size <= 0:
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = net(x_gpu)
        loss = hand_obj_dis(outputs, y_gpu, c, loss_weight, threshold=0.5)
        # loss_2 = loss_classification(outputs, y_gpu)
        #  = hand_obj_loss(outputs, y_gpu, threshold=threshold)
        # loss = criterion(outputs, y_gpu)

        loss.backward()
        optimizer.step()
    else:
        for batch_idx, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()  # zero the parameter gradients
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            # forward + backward + optimize
            outputs = net(x)
            # loss = hand_obj_loss(outputs, y, threshold=threshold )
            # loss = criterion(outputs, y)
            loss = hand_obj_dis(outputs, y_gpu, threshold=0.5)
            # loss_2 = loss_classification(outputs, y_gpu)
            #  = hand_obj_loss(outputs, y_gpu, threshold=threshold)
            # loss = criterion(outputs, y_gpu)
            # loss = loss_1 + loss_2 * alpha

            loss.backward()
            optimizer.step()

    # error_all.append(np.sqrt(loss.data.item())*max_dis_)
    dt = (time.time() - t0)
    if epoch % 100 == 0:
        dis_loss = np.sqrt(loss.data.item())
        print(epoch, ' dt:', "%0.4f" % dt, '  mean dis:', "%0.8f" % dis_loss)
        # print(epoch, ' dt:',"%0.4f" % dt, '  mean dis:', "%0.8f" % dis_loss,'classification loss', "%0.8f"%loss_2.data.item())
        if test_accuracy and epoch % 1000 == 0:
            test_num = 10000
            num_all = y_test_cpu.shape[0]
            assert test_num <= num_all
            s1 = np.random.randint(0, num_all - test_num)
            x_test = x_test_gpu[s1:s1 + test_num, :]
            y_test = y_test_cpu[s1:s1 + test_num, :]
            with torch.no_grad():
                outputs = net(x_test)
            outputs = outputs.cpu().numpy()
            validate(outputs, y_test, dis_scale, threshold=threshold)
        if loss < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'dis_scale': dis_scale,
                'obj_bounds': obj_bound,
                'hand_bounds': hand_joint_bounds,
                'layer_nums': layer_nums,
                'net_dims': [dim_in, dim_out]
            }, path_check_point)
            best_loss = loss
            # torch.save(net.state_dict(), path_save_model)
    epoch += 1
    if time.time() - t_start > 3600:
        break
    # if epoch > 100:
    #     break

# print('time cost for each epoch:', (time.time() - t_start) / epoch)

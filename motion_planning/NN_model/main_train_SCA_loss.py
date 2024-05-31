import time
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch.nn as nn
from nn_model import Net, loss_SCA,  load_dataset, validate

batch_size = 0
use_cuda = True
name = 'dataset10000000_15dis_sphere_2.npy'
data, num_s, max_dis = load_dataset(name)
max_dis_ = np.max(max_dis)

device = torch.device("cuda:0" if use_cuda else "cpu")
print('device:', device, '  data', data.shape)

num = int(data.shape[0] * 0.8)

test_num = data.shape[0] - num
x_train = torch.Tensor(data[:num, :num_s])
y_train = torch.Tensor(data[:num, num_s:])
dim_in = x_train.size(1)
dim_out = y_train.size(1)

x_test_gpu = torch.Tensor(data[num:, :num_s]).to(device)
y_test_cpu = data[num:, num_s:]


if batch_size <= 0:  # put all data into GPU for training
    x_gpu = x_train.to(device)
    y_gpu = y_train.to(device)
else:  # use DataLoader to send data to GPU in each batch_id
    dataset_train = TensorDataset(x_train, y_train)  # create dataset
    data_loader = DataLoader(dataset_train,
                             batch_size=batch_size,
                             num_workers=16,
                             pin_memory=True,
                             prefetch_factor=4,
                             persistent_workers=True,
                             drop_last=False,
                             )

net = Net(dim_in, dim_out)
net.to(device)

weight_p, bias_p = [], []
for name, p in net.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

import torch.optim as optim

# put all data to GPU memory


criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(),lr=0.01)
# optimizer = optim.Adam(net.parameters(),lr=0.01,weight_decay=0.001)
optimizer = optim.SGD([
    {'params': weight_p, 'weight_decay': 1e-5},
    {'params': bias_p, 'weight_decay': 0}],
    lr=0.1e-2, momentum=0.9)

resume = 1
test_accuracy = 1
# path_check_point = 'models/model_02.pt' # old loss
path_check_point = 'models/model_03.pt'  # modified loss, penalty output<0 and target>0
path_save_model = path_check_point[:-3] + '_eval'
epoch = 0
if resume:
    checkpoint = torch.load(path_check_point)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    net.train()

error_all = []
best_loss = 1
t_start = time.time()

while True:
    t0 = time.time()

    if batch_size <= 0:
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = net(x_gpu)
        loss = loss_SCA(outputs, y_gpu)
        # loss= criterion(outputs, y)
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
            loss = loss_SCA(outputs, y)
            # loss= criterion(outputs, y)

            loss.backward()
            optimizer.step()

    # error_all.append(np.sqrt(loss.data.item())*max_dis_)
    if epoch % 100 == 0:
        print(epoch, "%0.4f" % (time.time() - t0), np.sqrt(loss.data.item()) * max_dis_)
        if test_accuracy and epoch % 1000 == 0:
            test_num = 1000
            num_all = y_test_cpu.shape[0]
            assert test_num <= num_all
            s1 = np.random.randint(0, num_all - test_num)
            x_test = x_test_gpu[s1:s1 + test_num, :]
            y_test = y_test_cpu[s1:s1 + test_num, :]
            with torch.no_grad():
                outputs = net(x_test)
            outputs = outputs.cpu().numpy()
            validate(outputs, y_test, max_dis)
        if loss < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path_check_point)
            best_loss = loss
            # torch.save(net.state_dict(), path_save_model)
    epoch += 1

    # if epoch > 100:
    #     break

# print('time cost for each epoch:', (time.time() - t_start) / epoch)

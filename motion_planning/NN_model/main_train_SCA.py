import time
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys

# sys.path.append("..")
import numpy as np
import torch.nn as nn
# from NN_model.nn_model import Net, loss_SCA,  load_dataset, validate
# import NN_model.nn_model as nn_model
from nn_model import Net, loss_SCA_neg, load_dataset_SCA, validate_SCA_neg

batch_size = 0
use_cuda = True
name = 'allegro_SCA_2000000_01.npy'
add_cos_sin = 1
data, num_s, dis_scale = load_dataset_SCA(name, add_sin_cos=add_cos_sin, normalization=2, dec=0.5)
hand_joint_bounds = np.loadtxt('models/hand_joint_bound.txt')
# max_dis_ = np.max(max_dis)
# np.savetxt('models/dis_scale.txt', dis_scale)

device = torch.device("cuda:0" if use_cuda else "cpu")
print('device:', device, '  data', data.shape)

cv_num = 1
RMSE_train = np.zeros(cv_num)
RMSE_test = np.zeros(cv_num)
ACC = np.zeros([cv_num, 8])

for cv in range(cv_num):
    data_index = np.random.randint(0, data.shape[0], int(data.shape[0]))
    a = np.arange(0, data.shape[0], 1)
    data_index_test = np.setdiff1d(a, data_index)

    num = int(data.shape[0] * 0.5)

    # test_num = data.shape[0] - num
    x_train = torch.Tensor(data[data_index, :num_s])
    y_train = torch.Tensor(data[data_index, num_s:])
    dim_in = x_train.size(1)
    dim_out = y_train.size(1)

    x_test_gpu = torch.Tensor(data[data_index_test, :num_s]).to(device)
    y_test_cpu = data[data_index_test, num_s:]

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

    # put all data to GPU memory

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(),lr=0.01)
    optimizer = optim.AdamW([
        {'params': weight_p, 'weight_decay': 1e-5},
        {'params': bias_p, 'weight_decay': 0}],
        lr=1e-2)
    # optimizer = optim.SGD([
    #     {'params': weight_p, 'weight_decay': 1e-5},
    #     {'params': bias_p, 'weight_decay': 0}],
    #     lr=0.1e-2, momentum=0.9)

    #             y<0   0<y<1/2    y>1/2
    #  d <=0      w1      w2         1
    #  0<d<1/2    w3       1         1
    #  d>1/2      1        1         w1
    w1 = 0.1
    w2 = 2
    w3 = 2
    loss_weight = [w1, w2, w3]  #

    resume = 0
    test_accuracy = 1
    path_check_point = 'models/model_new_02.pt'  # modified loss, penalty output<0 and target>0
    epoch = 0
    if resume:
        checkpoint = torch.load(path_check_point)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.train()
        print('Reload model from', path_check_point)
    else:
        print('initialize a new model')

    error_all = []
    best_loss = 1
    t_start = time.time()

    while True:
        t0 = time.time()

        if batch_size <= 0:
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = net(x_gpu)
            loss = loss_SCA_neg(outputs, y_gpu, loss_weight, threshold=0.5)
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
                loss = loss_SCA_neg(outputs, y)
                # loss= criterion(outputs, y)

                loss.backward()
                optimizer.step()

        # error_all.append(np.sqrt(loss.data.item())*max_dis_)
        if epoch % 100 == 0:
            print('cv=', cv, '  epoch=', epoch, "%0.4f" % (time.time() - t0), np.sqrt(loss.data.item()))
            if test_accuracy and epoch % 1000 == 0:
                # test_num = 10000
                # num_all = y_test_cpu.shape[0]
                # assert test_num <= num_all
                # s1 = np.random.randint(0, num_all - test_num)
                # x_test = x_test_gpu[s1:s1 + test_num, :]
                # y_test = y_test_cpu[s1:s1 + test_num, :]
                x_test = x_test_gpu
                y_test = y_test_cpu
                with torch.no_grad():
                    outputs = net(x_test)
                outputs = outputs.cpu().numpy()
                validate_SCA_neg(outputs, y_test, dis_scale)
            if loss < best_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'dis_scale': dis_scale,
                    'hand_bounds': hand_joint_bounds,
                    'layer_nums': layer_nums,
                    'net_dims': [dim_in, dim_out]
                }, path_check_point)
                best_loss = loss
        epoch += 1

        if epoch > 10000:
            break

    with torch.no_grad():
        outputs = net(x_gpu)
    outputs = outputs.cpu().numpy()
    RMSE1, acc1 = validate_SCA_neg(outputs, data[data_index, num_s:], dis_scale, record=True)

    with torch.no_grad():
        outputs = net(x_test_gpu)
    outputs = outputs.cpu().numpy()
    RMSE2, acc2 = validate_SCA_neg(outputs, y_test, dis_scale, record=True)

    RMSE_train[cv] = RMSE1
    ACC[cv, 0:4] = acc1

    RMSE_test[cv] = RMSE2
    ACC[cv, 4:8] = acc2

save_result = path_check_point[:-3] + 'add_' + str(add_cos_sin)
np.savez(save_result, RMSE_train=RMSE_train, RMSE_test=RMSE_test, ACC=ACC,
         layer_nums=np.asarray(layer_nums),
         loss_weight=np.asarray(loss_weight))

print('========================save file at', save_result)
print('RMSE train:', np.mean(RMSE_train), 'std', np.std(RMSE_train))
print('collision accuracy for all', np.mean(ACC[:, 0]), 'overall', np.mean(ACC[:, 1]), np.std(ACC[:, 1]))
print('FP:', np.mean(ACC[:, 2]), np.std(ACC[:, 2]), "FN", np.mean(ACC[:, 3]), np.mean(ACC[:, 3]))

print('RMSE test:', np.mean(RMSE_test), 'std', np.std(RMSE_test))
print('collision accuracy for all', np.mean(ACC[:, 4]), 'overall', np.mean(ACC[:, 5]), np.std(ACC[:, 5]))
print('FP:', np.mean(ACC[:, 6]), np.std(ACC[:, 6]), "FN", np.mean(ACC[:, 7]), np.mean(ACC[:, 7]))

# print('time cost for each epoch:', (time.time() - t_start) / epoch)

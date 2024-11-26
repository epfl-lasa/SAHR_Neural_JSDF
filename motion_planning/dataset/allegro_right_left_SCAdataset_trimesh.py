import numpy as np
# import multiprocessing
import time
import tqdm
import trimesh

from allegro_collision import AllegroCollision
import sys

sys.path.append("../..")
import tools.rotations as rot


def dis(i):
    return allegro.calculate_dis_trimesh()


def cal_dis(num):
    from multiprocessing import Pool
    from contextlib import closing
    # Memory usage keep growing with Python's multiprocessing.pool
    # use this to close them
    with closing(Pool(20)) as a_pool:
        result = list(tqdm.tqdm(a_pool.imap(dis, range(num)), total=num))

    result = np.vstack(result)  # a list of array to array
    return result


right_hand = True
allegro = AllegroCollision(path_prefix='../../', use_22_dis=False, use_convex=True, use_trimesh=True, right=right_hand)

eps = 1e-8
num = 1000 * 2000

t0 = time.time()

dataset = cal_dis(num)

t1 = time.time() - t0

print(dataset.shape)
print('Total time cost:', t1, '   Average:', t1 / num)
dis_min = np.min(dataset[:, 19:], axis=1)
dis_min[dis_min < eps] = 0
print('obj-hand collision-free probability:', np.count_nonzero(dis_min) / dataset.shape[0])
dataset = np.float32(dataset)
if right_hand:
    suffix = '_03_test'
else:
    suffix = '_left_03'
name = 'allegro_SCA_' + str(num) + suffix

np.save(name, dataset)
print('save data to', name)

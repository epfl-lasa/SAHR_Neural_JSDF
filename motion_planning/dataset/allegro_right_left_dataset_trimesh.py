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
    q_rand = allegro.hand.generate_rand_joints(1)[0, :]
    link_poses = allegro.hand.get_all_links(q_rand, link_index=allegro.groups[g])

    index_choose = np.random.randint(0, len(link_poses))
    pose = link_poses[index_choose]

    mesh_num = allegro.groups[g][index_choose]
    min_cub = allegro.cuboid[mesh_num][0, :]
    max_cub = allegro.cuboid[mesh_num][1, :]
    x = np.array([np.random.uniform(min_cub[0], max_cub[0]),
                  np.random.uniform(min_cub[1], max_cub[1]),
                  np.random.uniform(min_cub[2], max_cub[2])])
    R = rot.quat2mat(pose[3:])
    x = R @ x + pose[:3]
    # meshes = []
    distances = np.zeros(len(allegro.groups[g]))
    for n, j in enumerate(allegro.groups[g]):
        mesh = trimesh.Trimesh.copy(allegro.trimesh_list[j])
        pose = link_poses[n]
        T = rot.pose2T(pose)
        mesh.apply_transform(T)
        # meshes.append(mesh)
        tri = trimesh.proximity.ProximityQuery(mesh)
        dis_tmp = tri.signed_distance(x.reshape(1, -1))
        assert dis_tmp.shape == (1,)
        distances[n] = - dis_tmp[0]

    data0 = np.hstack([q_rand, x, distances])
    return data0


def cal_dis(num):
    from multiprocessing import Pool
    from contextlib import closing
    # Memory usage keep growing with Python's multiprocessing.pool
    # use this to close them
    with closing(Pool(18)) as a_pool:
        result = list(tqdm.tqdm(a_pool.imap(dis, range(num)), total=num))

    # result = a_pool.map(dis, range(num))
    result = np.vstack(result)  # a list of array to array
    result = result[~np.isnan(result).any(axis=1)]  # remove rows with nan
    return result


eps = 1e-8
right_hand = False
# [0,1,4] ,since 1 2 3 fingers are the same.
for g in [1, 4]:  # [palm and amount, index, thumb] #   # Order: base, index, middle, ring, thumb
    data = []
    rate_list = [1.1, 2, 5]
    points_num = []
    for rate in rate_list:
        allegro = AllegroCollision(path_prefix='../../', use_22_dis=False, use_convex=True, rate=rate, use_trimesh=True,
                                   right=right_hand)

        print('start g=', g, 'rate', rate, 'link_index', allegro.groups[g])
        for i in allegro.groups[g]:
            print(allegro.hand.all_links[i])
        t0 = time.time()

        # torch.multiprocessing.set_start_method('spawn')
        # paramlist = list(itertools.product(range(num), range(x_obj_num) ))
        num = 1000 * 1000 * 2
        # num = 10000
        dataset = cal_dis(num)

        t1 = time.time() - t0

        print(dataset.shape)
        print('Total time cost:', t1, '   Average:', t1 / num)

        dis_min = np.min(dataset[:, 19:], axis=1)
        dis_min[dis_min < eps] = 0
        print('obj-hand collision-free probability:', np.count_nonzero(dis_min) / dataset.shape[0])
        dataset = np.float32(dataset)
        if right_hand:
            name = 'data/obj_single_convex_' + str(g) + '_rate' + str(int(rate // 1))
        else:
            name = 'data/obj_single_convex_left_' + str(g) + '_rate' + str(int(rate // 1))
        np.save(name, dataset)

        # data.append(dataset)


        #
        # a_nonzero = np.nonzero(dis_min)[0]
        # a_zero = np.where(dis_min == 0)[0]
        #
        # if rate == rate_list[0]:
        #     a_choice = np.random.choice(a_nonzero, int(len(a_zero) / 2), replace=False)  # choose with nums of zeros
        #     half = np.concatenate([a_zero, a_choice])
        #     half = np.sort(half)
        #     dataset_half = dataset[half, :]
        # else:
        #     dataset_half = dataset[a_nonzero, :]
        #
        # dis_min2 = np.min(dataset_half[:, 19:], axis=1)

        # data_dis = dataset_half[:,19:]
        # data_dis[data_dis<1e-5] = 0
        # dataset_half[:, 19:] = data_dis

        # name = 'obj_single_convex_' + str(g) + '_rate' + str(int(rate // 1))
        # dataset_half = np.float32(dataset_half)
        # print(dataset_half.shape)
        # data.append(dataset_half)
        # points_num.append(int(len(a_zero) / 2))



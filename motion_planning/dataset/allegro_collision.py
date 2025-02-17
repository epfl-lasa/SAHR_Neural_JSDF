import time

import numpy as np
import pymesh as pm
import fcl
import trimesh
import sys

sys.path.append("../..")
sys.path.append("../")
import kinematics.allegro_hand as allegro
import tools.rotations as rot


class AllegroCollision:
    def __init__(self, path_prefix='../', use_22_dis=False, use_convex=True, rate=1.1, use_trimesh=False, right=True):
        self.hand = allegro.Robot(use_fingers=[1, 1, 1, 1], path_prefix=path_prefix, all_link_fk=True, meshes=True, right_hand=right)
        if right:
            all_link_meshes = ['allegro_mount', 'base_link', 'link_0.0', 'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip',
                            'link_0.0', 'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip', 'link_0.0', 'link_1.0',
                            'link_2.0', 'link_3.0', 'link_3.0_tip', 'link_12.0_right', 'link_13.0', 'link_14.0',
                            'link_15.0', 'link_15.0_tip']
        else:
            # This is a list of name for meshes.
            all_link_meshes = ['allegro_mount', 'base_link_left', 'link_0.0','link_1.0','link_2.0','link_3.0','link_3.0_tip','link_0.0','link_1.0','link_2.0','link_3.0','link_3.0_tip', 'link_0.0','link_1.0','link_2.0','link_3.0','link_3.0_tip', 'link_12.0_left', 'link_13.0', 'link_14.0','link_15.0', 'link_15.0_tip']
        assert len(self.hand.all_links) == len(all_link_meshes)

        # generate rand pose for the object
        obj_range_lb = [-0.07, -0.2, -0.1]  # x_min, y_min, z_min, RPY
        obj_range_ub = [0.3, 0.3, 0.4]  # x_min, y_min, z_min, RPY
        obj_bound = np.array([obj_range_lb, obj_range_ub])
        np.savetxt('obj_bound.txt', obj_bound, delimiter=' ')

        # x_obj = np.zeros([nums, 3])
        # for i in range(3):
        #     x_tmp = np.random.uniform(obj_range_lb[i], obj_range_ub[i], nums)
        #     x_obj[:, i] = x_tmp

        box = fcl.Sphere(0.001)

        # path of allegro
        path = path_prefix + 'description/allegro_all/meshes/'
        meshes = []
        meshes_tri = []
        for i, name in enumerate(all_link_meshes):
            if use_convex:
                name = name + '_convex'
                suffix = '.stl'
            else:
                suffix = '.STL' if i else '.stl'
            mesh_path = path + name + suffix
            # if use_convex:
            #     print(mesh_path)
            mesh = pm.meshio.load_mesh(mesh_path)
            if use_trimesh:
                mesh2 = trimesh.load(mesh_path)
                meshes_tri.append(trimesh.Trimesh.copy(mesh2))
            meshes.append(mesh)

        fcl_mesh = []
        cuboid = []
        for mesh in meshes:
            fcl_mesh_ = fcl.BVHModel()
            fcl_mesh_.beginModel(len(mesh.vertices), len(mesh.faces))
            fcl_mesh_.addSubModel(mesh.vertices, mesh.faces)
            fcl_mesh_.endModel()
            min_cub = np.min(mesh.vertices, axis=0)
            max_cub = np.max(mesh.vertices, axis=0)
            cuboid.append(cuboid_enlarge(min_cub, max_cub, rate))
            fcl_mesh.append(fcl_mesh_)
        if use_22_dis is False:
            self.groups = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16],
                           [17, 18, 19, 20, 21]]  # base, index, middle, ring, thumb
        else:
            self.groups = list(range(22))

        self.use_22_dis = use_22_dis
        self.group_num = len(self.groups)

        self.objs = [fcl.CollisionObject(fcl_mesh[i], fcl.Transform()) for i in
                     range(len(self.hand.all_links))]  # initialize all bodies of the self.hand
        self.objs += [fcl.CollisionObject(box, fcl.Transform())]  # add the box
        self.req = fcl.DistanceRequest()
        self.res = fcl.DistanceResult()

        self.cuboid = cuboid
        if use_trimesh:
            self.trimesh_list = meshes_tri

    def calculate_dis_trimesh(self, q=None):
        ### 1 rand q
        # forward kinematics
        if q is None:
            np.random.seed()  # reset the seed for multi processing, otherwise they will have the same results form random
            q = self.hand.generate_rand_joints(1)[0]
        link_poses = self.hand.get_all_links(q)
        ###
        ### 2 for loop for each pair
        ### 3 initialize two managers, add mesh
        ## collision detect, if no collision, get distance.
        dis_all = []
        for j in range(self.group_num):
            manager_1 = trimesh.collision.CollisionManager()
            for mesh_index in self.groups[j]:
                T = rot.pose2T(link_poses[mesh_index])
                manager_1.add_object(str(mesh_index), self.trimesh_list[mesh_index], transform=T)
            for k in range(j + 1, self.group_num):
                if j == 0 and k in [1, 2, 3, 4]:
                    choose_group = self.groups[k][2:]  # remove the first two meshes of fingers
                else:
                    choose_group = self.groups[k]
                manager_2 = trimesh.collision.CollisionManager()
                for mesh_index in choose_group:
                    T = rot.pose2T(link_poses[mesh_index])
                    manager_2.add_object(str(mesh_index), self.trimesh_list[mesh_index], transform=T)

                is_collision, contact_data = manager_1.in_collision_other(manager_2, return_data=True)
                if is_collision:
                    depth = [c.depth for c in contact_data]
                    dis = - max(depth)
                else:
                    dis = manager_1.min_distance_other(manager_2)
                    assert dis > 0
                dis_all.append(dis)
        assert len(dis_all) == 10
        data = np.hstack([q, np.asarray(dis_all)])
        return data

    def calculate_dis_single(self, q, x_obj, g=0):
        assert q.shape == (16,)
        nums = len(self.groups[g])

        # update pose 
        poses = self.hand.get_all_links(q)
        finger = [self.objs[i] for i in self.groups[g]]
        for i, num in enumerate(self.groups[g]):
            tf1 = fcl.Transform(poses[num][3:], poses[num][:3])  # Quaternion rotation and translation
            finger[i].setTransform(tf1)

        tf2 = fcl.Transform(x_obj)
        obj = self.objs[-1]
        obj.setTransform(tf2)

        dist = [fcl.distance(finger[i],
                             obj,
                             self.req, self.res) for i in range(nums)]

        return np.asarray(dist)

    def calculate_dis(self, q, x_obj=None, nearest_points=False, hand_obj=1):
        """

        hand_obj = 1   only return dis between hands, 10
                    2                 hand and obj      5
                    3         all                     15
        """
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        assert q.shape[1] == 16

        dis_all = []
        nearest_ps = []
        for i in range(q.shape[0]):
            if x_obj is None:
                x_obj = np.array([1, 1, 1.])

            manager_all = [fcl.DynamicAABBTreeCollisionManager() for i_tmp in
                           range(len(self.groups) + 1)]  # to make them different, do not use list*6

            poses = self.hand.get_all_links(q[i, :])
            # poses = self.hand.get_all_links(qh0)
            for j, pose in enumerate(poses):
                tf1 = fcl.Transform(pose[3:], pose[:3])  # Quaternion rotation and translation
                self.objs[j].setTransform(tf1)  # update pose for the j-th mesh
            tf2 = fcl.Transform(x_obj)
            self.objs[-1].setTransform(tf2)

            # generate groups for managers
            for j, g in enumerate(self.groups):
                if self.use_22_dis is False:
                    obj_group = [self.objs[k] for k in g]
                else:
                    obj_group = [self.objs[g]]
                manager_all[j].registerObjects(obj_group)
            manager_all[-1].registerObjects([self.objs[-1]])
            _ = [manager.setup() for manager in manager_all]

            # start to compute the dis
            dis = []
            # dis_hand = []
            req = fcl.DistanceRequest()
            if nearest_points:
                nearest_p = []
            for j in range(len(manager_all) - 1):
                for k in range(j + 1, len(manager_all)):
                    if k == len(manager_all) - 1 and hand_obj == 1:  # only hand
                        continue
                    if k != len(manager_all) - 1 and hand_obj == 2:  # only hand and obj
                        continue
                    rdata = fcl.DistanceData(request=req)
                    if j == 0 and k in [1, 2, 3, 4]:  # base and the 4 fingers
                        manager_1 = fcl.DynamicAABBTreeCollisionManager()
                        manager_2 = fcl.DynamicAABBTreeCollisionManager()
                        g1 = [self.objs[l] for l in self.groups[0]]
                        g2 = [self.objs[l] for l in self.groups[k][2:]]  # remove the first two links, because they are
                        # connected with the palm.
                        manager_1.registerObjects(g1)
                        manager_2.registerObjects(g2)
                        manager_1.setup()
                        manager_2.setup()
                        manager_1.distance(manager_2, rdata, fcl.defaultDistanceCallback)  # 95.3 µs
                    else:
                        manager_all[j].distance(manager_all[k], rdata, fcl.defaultDistanceCallback)  # 160 µs
                    # print('Collision between manager 1 and manager 2?: {}'.format(rdata.result.))
                    dis.append(rdata.result.min_distance)
                    if nearest_points:
                        nearest_p.append(np.asarray(rdata.result.nearest_points))
                    # if k !=len(manager_all)-1:
                    #     dis_hand.append(rdata.result.min_distance)
            # min_dis = min(dis) # This would lose much information
            # min_dis_hand = min(dis_hand)
            dis_15 = np.asarray(dis)
            dis_15 = np.clip(dis_15, 0, 10)
            dis_all.append(dis_15)
            if nearest_points:
                nearest_ps.append(nearest_p)
            # if min_dis<=0:
            #     min_dis=0
            # if min_dis_hand<=0:
            #     min_dis_hand=0
            # dis_min2_15 = np.concatenate([np.array([min_dis, min_dis_hand]), dis_15])
            # dis_all.append(dis_min2_15)

        # dis_all = np.asarray(dis_all)
        # if hand_obj==1:
        #     index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 12]
        # elif hand_obj==2:
        #     index = [4, 8, 11, 13, 14]  # only dis between obj
        # else:
        #     index = list(range(15))  # all distance

        if nearest_points:
            return dis_all, nearest_points
        else:
            return dis_all


def cuboid_enlarge(min_cub, max_cub, rate):
    center = np.array([(max_cub[i] + min_cub[i]) / 2 for i in range(3)])
    length = np.array([(max_cub[i] - min_cub[i]) / 2 for i in range(3)]) * rate
    min_cub_m = center - length
    max_cub_m = center + length
    return np.vstack([min_cub_m, max_cub_m])

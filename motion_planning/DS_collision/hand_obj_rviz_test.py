import numpy as np
import rospy
import time
# import sys
# sys.path.append('../')

# run this and under `description/allegro-all/` run `roslaunch urdf_tutorial display_allegro.launch` to launch Rviz, `keyboard_cmd.py` for controlling the obstacle
# run current script, in Rviz add visualize_cloud, set world base as `base`. Then control the obstacle by keyboard:
# A S D W I K
# This is to show how the obstacle will affect the movement of the hand

# roslaunch urdf_tutorial display.launch model:=allegro_right_mount_convex_meshes.urdf for controlling joints by mouse

from Rviz_visualize import obj_Rviz, hand_Rviz, moving_obj_ros
from DS import linear_system, Modulation
from obstacle import point_obstacle

hand_visualize = hand_Rviz()
obj_visualize = obj_Rviz(init=False)

# x0 = np.array([[0.07, 0.044, 0.2]])
x0 = np.array([[0.07, 0.044, 0.17]])
x0 = np.array([[7.90000000e-02,  0, 1.97000000e-01]])
num = 1
x_obj = np.repeat(x0, num, axis=0)
x_obj[:, 1] = np.linspace(0.04, 0.06, num)
x_obj[:, 1] = 0
obj = moving_obj_ros(init=False, x_obj=x_obj, dt=0.01)

g = [0, 1, 2, 3, 4]  # here we use full fingers, with 16D DS modulation
if 0 in g:
    dim = (len(g) - 1) * 4
else:
    dim = len(g) * 4
obs = point_obstacle(g=g, use_cuda=True)

q0 = np.zeros(16)
q0[12] = 0.5
hand_visualize.send_cmd(q0)

q = np.copy(q0)
q[5:8] = np.array([0.8, 0.8, 0])
ds_0 = linear_system(q)

whole_hand = False
# only use the min distance/grad between the obj and the whole hand? then the grad is zero for non-min fingers.
# or the distance/grad for each finger?
# Then I need 4 M matrix and combine then together. diag(M_i, M_m, M_r, M_t) @ dq
if whole_hand:   # use whole hand  and 16D
    modify_DS = Modulation(dim)
else:  # use 4 DS for 4 fingers, respectively.
    modify_DS = [Modulation(4) for i in range(dim // 4)]

# obstacle


dis, grad = obs.dis_and_grad(q0, x_obj)

r = rospy.Rate(100)
dt = 0.01

q_now = np.copy(q0)
safety_margin = 0.01  # meter
dis_scale = 100
q_free = np.load('index_collision.npy')
q_free = q_free[:, 1:] + np.array([0.5,0,0])

q_sim_0 = np.copy(q0)
q_sim_0[4:8] = np.array([0, 0.25, 0.25, 0])

q_sim_1 = np.copy(q0)
q_sim_1[4:8] = np.array([0, 1, 0.25, 0])

t0 = time.time()
while not rospy.is_shutdown():
    dq = ds_0.eval(q_now)
    #
    t1 = time.time()
    dis, grad = obs.dis_and_grad(q_now, obj.x, whole_hand=whole_hand, real_dis=True)
    t2 = time.time() - t1

    # print(t2, grad)
    if whole_hand: 
        gamma = (dis - safety_margin) * dis_scale + 1
        M = modify_DS.get_M(grad, gamma,dq=dq, rho=5)
        dq = M @ dq
        if dis < 0.9 * safety_margin:
            dq = 0.2 * grad / np.linalg.norm(grad)
    else:
        gamma = [(d - safety_margin) * dis_scale + 1 for d in dis]
        M = np.zeros((16, 16))
        for i, modify_ in enumerate(modify_DS):
            M[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = modify_.get_M(grad[i], gamma[i], dq=dq[i * 4:(i + 1) * 4], rho=5)
        dq = M @ dq

        for i in range(4):
            if dis[i] < 0.9 * safety_margin:
                dq[i*4:(i+1)*4] = 0.2 * grad[i] / np.linalg.norm(grad[i])

    # q_now += dq * dt
    if int(time.time() - t0) % 2:
        q_now = q_sim_0
    else:
        q_now = q_sim_1
    # hand_visualize.send_cmd(q_now, dq)
    hand_visualize.send_cmd(q_now)
    obj_visualize.send_cmd(obj.x)
    print(obj.x)
    # obj_visualize.send_cmd(q_free)


    r.sleep()

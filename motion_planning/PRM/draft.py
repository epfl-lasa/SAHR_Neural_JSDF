import numpy as np

from PRM_tools import collision_check

x_obj = np.array([[8.10000000e-02, 0, 1.87000000e-01]])
use_cuda = False

a = collision_check(x_obj, use_cuda=use_cuda)

q = np.random.uniform(a.nn.hand_bound[0, :], a.nn.hand_bound[1, :], size=(20, 16))
s_goal = (-0.19693, 1.3903904, 1.3273159, 0.35897200000000007,
          -0.06984199999999996, 1.3148996000000002, 1.27591, 0.43,
          0.0, 1.3293476000000002, 1.2510544000000001, 0.55,
          1.1357499,      0.9659528, 1.5200892, 0.6767379)
# q = np.array(s_goal)
# output = a.SCA_eval(q)

dis = a.get_dis(q)
print(dis)

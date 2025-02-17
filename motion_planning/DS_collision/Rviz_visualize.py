import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Twist


class hand_Rviz:
    def __init__(self, init=True):
        if init:
            rospy.init_node('DS_dynamical_env')
        self.robot_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        msg = JointState()
        msg.name = np.asarray(['joint_' + str(i) for i in range(16)])
        self.msg = msg
        self.msg.position = np.zeros(16)
        self.msg.velocity = np.zeros(16)

    def send_cmd(self, q, dq=None):
        self.msg.header.stamp = rospy.Time.now()
        self.msg.position = q
        if dq is not None:
            self.msg.velocity = dq
        self.robot_state_pub.publish(self.msg)


class obj_Rviz:
    def __init__(self, init=True, frame='base'):
        if init:
            rospy.init_node('obj_obstacle')
        self.pub = rospy.Publisher('obj_visualize', PointCloud, queue_size=10)
        self.frame = frame

    def send_cmd(self, x: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        clouds = PointCloud()
        clouds.header.frame_id = self.frame
        # p = Point32()
        # clouds.points = x.tolist()
        for i in range(x.shape[0]):
            clouds.points.append(Point32(x[i, 0], x[i, 1], x[i, 2]))
        self.pub.publish(clouds)


class moving_obj_ros:
    def __init__(self, init=True, x_obj=np.zeros(3), dt=0.001):
        if init:
            rospy.init_node('update_obs_position')
        self.sub = rospy.Subscriber("/keyboard_cmd", Twist, self.obj_vel_cb, queue_size=10)
        self.x = x_obj
        self.dt = dt

    def obj_vel_cb(self, state: Twist):
        vel = np.array([state.linear.x, state.linear.y, state.linear.z])
        self.x += vel * self.dt

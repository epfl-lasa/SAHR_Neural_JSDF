# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import time
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

X_dimensions = np.array([(-100, 100), (-100, 100), (0, 100)])  # dimensions of Search Space
# obstacles
# Obstacles = np.array(
#     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#      (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
file_path = '/home/xiao/research/lasa/iiwa_allegro_sim/motion_planning/DS_collision/'
data = np.load(file_path + 'rrt_data.npz')

X_dimensions = data['space']
Obstacles = data['obs']
x_init = tuple(data['start'])
x_goal = tuple(data['goal'])
# x_init = (0, 0, 0)  # starting location
# x_goal = (100, 100, 100)  # goal location
# x_init = (21.10749584, 35.09713897, 31.81533276)
# x_goal = (88.35887922, 61.13615544, 40.44495946)


Q = np.array([(0.05, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
r = (X_dimensions[0,1] - X_dimensions[0,0]) / 100/2
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
t0 = time.time()
path = rrt.rrt_star()
t1 = time.time() - t0
print(t1)
path = np.array(path)
np.savetxt(file_path + 'rrt_star_path.txt', path)
# plot
plot = Plot("rrt_star_3d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)

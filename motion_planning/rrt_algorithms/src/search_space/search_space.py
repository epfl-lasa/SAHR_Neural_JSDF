# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, get_dis):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        self.get_dis = get_dis
        # if O is None:
        #     self.obs = index.Index(interleaved=True, properties=p)
        # else:
        #     # r-tree representation of obstacles
        #     # sanity check
        #     if any(len(o) / 2 != len(dimension_lengths) for o in O):
        #         raise Exception("Obstacle has incorrect dimension definition")
        #     if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
        #         raise Exception("Obstacle start must be less than obstacle end")
        #     self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        # return self.obs.count(x) == 0
        return self.get_dis(x, x_obj = self.x_obj) > 0

    def sample_free(self, grad=False):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            # x[4:12] = np.zeros(8)
            if grad:
                d, g = self.get_dis(x, gradient=True, x_obj = self.x_obj)
                if d >= 0:
                    return d, g
            else:
                if self.obstacle_free(x):
                    return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        # points = es_points_along_line(start, end, r)
        # coll_free = all(map(self.obstacle_free, points))
        points = straight_line(start, end, r)
        coll_free = self.obstacle_free(points)
        return coll_free

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        # return tuple(x)
        return x


def straight_line(start, end, r):
    d = np.linalg.norm(np.array(start) - np.array(end))
    n = int(np.ceil(d / r))
    return np.linspace(np.array(start), np.array(end), n)

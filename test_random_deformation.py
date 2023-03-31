import numpy as np
import matplotlib.pyplot as plt

import plt_utils
from pyshapetree import ShapeTree
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    line, = ax.plot([], [])  # empty line
    ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    linebuilder = plt_utils.LineBuilder(line, 0.05)
    plt.show()
    pts = np.vstack(line.get_xydata()).T
    shape_tree = ShapeTree(pts)

    # plot with noise applied to bookstein coordinates
    sigma = 1
    coords = shape_tree.get_global_coords(pts[:, [0]], pts[:, [-1]],
                                          alter_midpoint_bookstein_callback=lambda pt: pt + .1*np.array([[np.random.normal(0, sigma), np.random.normal(0, sigma)]]).T)
    xs, ys = np.split(coords, 2, 0)
    xs, ys = xs.flatten(), ys.flatten()
    fig, ax = plt.subplots()
    line, = ax.plot(xs, ys)
    ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    ax.set_title('Noisy')

    plt.show()


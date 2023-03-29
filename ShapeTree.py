import numpy as np

import matplotlib.pyplot as plt

import bookstein
import plt_utils

class ShapeTree:
    class _ShapeTreeNode:
        def __init__(self, subcurve_start_point, subcurve_end_point, subcurve_midpoint):
            self.left = None
            self.right = None

            self.subcurve_start_point = subcurve_start_point
            self.subcurve_end_point = subcurve_end_point

            self.c, self.A, self.b = bookstein.get_bookstein_transformation_2d(self.subcurve_start_point, self.subcurve_end_point)

            self.midpoint_bookstein = self.c * (self.A @ (subcurve_midpoint - self.b))
    
    # points takes a 2 x n matrix
    def __init__(self, points):
        self.points = points
        if len(points) % 2 == 1:
            raise ValueError("Must pass in an odd number of points for binary tree structure")
        self.root = self.create_subtree(0, points.shape[1] - 1)

    # subcurve_start/end indexes self.points
    def create_subtree(self, subcurve_start, subcurve_end):
        midpoint = (subcurve_start + subcurve_end) // 2
        ret = self._ShapeTreeNode(self.points[:, [subcurve_start]], self.points[:, [subcurve_end]], self.points[:, [midpoint]])
        if midpoint == subcurve_start + 1:  # leaf node
            return ret
        ret.left = self.create_subtree(subcurve_start, midpoint)
        ret.right = self.create_subtree(midpoint, subcurve_end)
        return ret

    # callback is called on every bookstein coordinate with point as arg
    def plot(self, callback):
        coords = [self.root.subcurve_start_point, self.root.subcurve_end_point]

        # pre-order dfs
        node_stack = [self.root]
        while len(node_stack) != 0:
            curr_node = node_stack.pop()
            if curr_node.left is not None:
                node_stack.append(curr_node.left)
            if curr_node.right is not None:
                node_stack.append(curr_node.right)
            midpoint_bookstein = callback(curr_node.midpoint_bookstein)

            # we transform the bookstein midpoint back to global coords by reversing the transformation
            c_inv, A_inv, b = curr_node.c**-1, np.linalg.inv(curr_node.A), curr_node.b
            midpoint = c_inv * (A_inv @ midpoint_bookstein) + b
            coords.append(midpoint)

        coords = np.hstack(coords)
        xs, ys = np.split(coords, 2, 0)
        xs, ys = xs.flatten(), ys.flatten()
        return plt_utils.plot_coordinates(xs, ys, (xs.min() - 1, xs.max() + 1), (ys.min() - 1, ys.max() + 1))

if __name__ == "__main__":
    pts = np.array([[1, 2, 3, 4, 5],
                    [5, 4, 3, 2, 1]])
    
    xs, ys = np.split(pts, 2, 0)
    xs, ys = xs.flatten(), ys.flatten()
    inputted = plt_utils.plot_coordinates(xs, ys, (xs.min() - 1, xs.max() + 1), (ys.min() - 1, ys.max() + 1))
    inputted.gca().set_title("Input to shape tree")

    shape_tree = ShapeTree(pts)
    recovered = shape_tree.plot(callback=lambda pt: pt + np.array([[np.random.normal(0, 1), np.random.normal(0, 1)]]).T)
    recovered.gca().set_title("Recovered from shape tree")
    plt.show()
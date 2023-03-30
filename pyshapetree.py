import numpy as np

import bookstein

class ShapeTree:
    class _ShapeTreeNode:
        def __init__(self, subcurve_start_point, subcurve_end_point, subcurve_midpoint):
            self.left = None
            self.right = None

            c, A, b = bookstein.get_bookstein_transformation_2d(subcurve_start_point, subcurve_end_point)
            self.c, self.A, self.b = bookstein.get_bookstein_transformation_2d(subcurve_start_point, subcurve_end_point)
            self.midpoint_bookstein = c * (A @ (subcurve_midpoint - b))
    
    # points takes a 2 x n matrix
    def __init__(self, points):
        self.points = points
        self.root = self.create_subtree(0, points.shape[1] - 1)

    # subcurve_start/end indexes self.points
    def create_subtree(self, subcurve_start, subcurve_end):
        midpoint = (subcurve_start + subcurve_end) // 2
        ret = self._ShapeTreeNode(self.points[:, [subcurve_start]], self.points[:, [subcurve_end]], self.points[:, [midpoint]])
        if midpoint - subcurve_start > 1:
            ret.left = self.create_subtree(subcurve_start, midpoint)
        if subcurve_end - midpoint > 1:
            ret.right = self.create_subtree(midpoint, subcurve_end)
        return ret
    
    # in-order dft
    @staticmethod
    def traverse_subtree(sub_root, subcurve_start_point, subcurve_end_point, callback=None):
        if sub_root is None:
            return []
        c, A, b = bookstein.get_bookstein_transformation_2d(subcurve_start_point, subcurve_end_point)
        c_inv, A_inv = c**-1, np.linalg.inv(A)

        midpoint_bookstein = sub_root.midpoint_bookstein
        if callback is not None:
            midpoint_bookstein = callback(midpoint_bookstein)

        # we transform the bookstein midpoint back to global coords by inverting the transformation
        midpoint = c_inv * (A_inv @ midpoint_bookstein) + b

        ret = []
        ret.extend(ShapeTree.traverse_subtree(sub_root.left, subcurve_start_point, midpoint))
        ret.append(midpoint)
        ret.extend(ShapeTree.traverse_subtree(sub_root.right, midpoint, subcurve_end_point))
        
        return ret

    # callback is called on every bookstein coordinate with point as arg
    # returns 2 x n matrix
    # reconstructs the curve relative to first and last points
    def get_global_coords(self, first_point, last_point, callback=None):
        coords = self.traverse_subtree(self.root, first_point, last_point, callback)
        return np.hstack([first_point] + coords + [last_point])
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import plt_utils

    # test that we can recover points from shape tree
    num_pts = np.random.randint(10, 16)
    print(f"Number input points: {num_pts}")
    pts = np.array([[np.random.random()*20 - 10 for _ in range(num_pts)],
                    [np.random.random()*20 - 10 for _ in range(num_pts)]])
    
    xs, ys = np.split(pts, 2, 0)
    xs, ys = xs.flatten(), ys.flatten()
    graph = plt_utils.plot_coordinates(xs, ys, (xs.min() - 1, xs.max() + 1), (ys.min() - 1, ys.max() + 1))

    shape_tree = ShapeTree(pts)
    xs, ys = np.split(shape_tree.get_global_coords(pts[: , [0]], pts[: , [-1]]), 2, 0)
    xs, ys = xs.flatten(), ys.flatten()
    graph = plt_utils.plot_coordinates(xs, ys, (xs.min() - 1, xs.max() + 1), (ys.min() - 1, ys.max() + 1), graph, marker="x")
    
    plt.legend(markerfirst=True, labels=["Inputted", "Recovered"])
    graph.gca().set_title("Test Point Recovery")

    plt.show()
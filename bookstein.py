import numpy as np

# returns c, A, b such that cA(X - b) maps pt1 to (-0.5, 0) and pt2 to (0.5, 0)
# pt1 and pt2 should be 2d column vectors
# c is scaling factor
# A is 2d rotation matrix
# b is translation, column vector
# https://people.stat.sc.edu/dryden/STAT718A/notes/shape-chap2A.pdf
def get_bookstein_transformation_2d(pt1, pt2):
    b = np.array([[1/2 * (pt1[0] + pt2[0]).item(), 1/2 * (pt1[1] + pt2[1]).item()]]).T
    diff = pt2 - pt1
    theta = -np.arctan2(diff[1].item(), diff[0].item())
    A = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    # theta = -np.arctan(diff[1].item() / diff[0].item())
    sqr_dist = (diff[0].item())**2 + (diff[1].item())**2
    c = (sqr_dist)**(-1/2) if sqr_dist != 0 else 0
    return c, A, b

if __name__ == "__main__":
    import plt_utils
    import matplotlib.pyplot as plt

    pt1 = np.array([[3, 9]]).T
    pt2 = np.array([[22, -18]]).T
    pts = np.hstack((pt1, pt2))
    
    xs, ys = np.split(pts, 2, 0)
    before = plt.figure(plt_utils.plot_coordinates(xs.flatten(), ys.flatten(), (-30, 30), (-30, 30)))

    c, rot_mat, b = get_bookstein_transformation_2d(pt1, pt2)
    print(f"Scale: {c}")
    print(f"Rotate :\n{rot_mat}")
    print(f"Translate:\n{b}")
    bs = np.tile(b, pts.shape[1])

    translated_pts = pts - bs
    xs, ys = np.split(translated_pts, 2, 0)
    translated = plt.figure(plt_utils.plot_coordinates(xs.flatten(), ys.flatten(), (-30, 30), (-30, 30)))
    translated.gca().set_title("Translated")

    rot_pts = (rot_mat @ (pts - bs))
    xs, ys = np.split(rot_pts, 2, 0)
    rotated = plt.figure(plt_utils.plot_coordinates(xs.flatten(), ys.flatten(), (-30, 30), (-30, 30)))
    rotated.gca().set_title("Rotated")

    scaled_pts = c * (rot_mat @ (pts - bs))
    
    xs, ys = np.split(scaled_pts, 2, 0)
    scaled = plt.figure(plt_utils.plot_coordinates(xs.flatten(), ys.flatten(), (-2, 2), (-2, 2)))
    scaled.gca().set_title("Scaled")

    plt.show()

# returns square procustes distance
def sqr_procrustes(A, B):
    return np.sum(np.square(A-B))

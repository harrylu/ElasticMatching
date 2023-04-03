import numpy as np

import bookstein
import pyshapetree

# A, B are 2 x N matrix
# returns cost of deforming A to B
def match(A, B):

    # depth-first post-order
    def algorithm(sub_root_A, depth):
        B_num_pts = B.shape[1]
        table = np.zeros((B_num_pts, B_num_pts))

        # base case; no node, cost is zero
        if sub_root_A is None:
            return table
        
        # repetitive code, fix later
            
        # base case; sub_root is leaf
        if sub_root_A.left is None and sub_root_A.right is None:
            for e in range(B_num_pts):
                for s in range(e - 1):
                    # find min midpoint on b
                    costs = []
                    for i in range(s + 1, e):
                        dif = bookstein.sqr_procrustes(sub_root_A.midpoint_bookstein, B_bookstein_table[s, e, i])
                        weight = 1 / (2**depth) * 1e3
                        costs.append(weight * dif)
                        #assert weight*dif > 0
                    table[s, e] = min(costs)
                    #print(table[s, e])
                    # assert table[s, e] != 0
            # assert table[1, B_num_pts - 1] != 0
            #print(table)
            #input()
            return table
        
        # get tables of children
        left_table = algorithm(sub_root_A.left, depth + 1)
        right_table = algorithm(sub_root_A.right, depth + 1)
        # make new table by going through all possible combinations from tables of children for each entry
        for e in range(B_num_pts):
            for s in range(e - 1):
                # find min midpoint on b
                costs = []
                for i in range(s + 1, e):
                    left_cost = left_table[s, i]
                    right_cost = right_table[i, e]
                    dif = bookstein.sqr_procrustes(sub_root_A.midpoint_bookstein, B_bookstein_table[s, e, i])
                    weight = 1 / (2**depth) * 1e3
                    costs.append(left_cost + right_cost + weight * dif)
                    #assert left_cost + right_cost + weight * dif > 0
                table[s, e] = min(costs)
                #print(table[s, e])
                # assert table[s, e] != 0
        assert table[1, B_num_pts - 1] != 0
        return table
    
    A_shape_tree = pyshapetree.ShapeTree(A)

    B_num_pts = B.shape[1]
    # B_bookstein_table[s, e, i] stores bookstein coordinate of ith point of B (B_i) relative to B_s and B_e for s <= e
    B_bookstein_table = np.empty((B_num_pts, B_num_pts, B_num_pts), dtype=object)
    for e in range(B_num_pts):
        for s in range(e - 1):
            c, A, b = bookstein.get_bookstein_transformation_2d(B[:, [s]], B[:, [e]])
            for i in range(s + 1, e):
                ith_point_bookstein = c * (A @ (B[:, [i]] - b))
                B_bookstein_table[s, e, i] = ith_point_bookstein

    return algorithm(A_shape_tree.root, 0)

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import plt_utils
    # test with self-draw
    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    line1, = ax.plot([], [])  # empty line
    ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    linebuilder = plt_utils.LineBuilder(line1, 0.1)
    plt.show()
    line1_data = line1.get_xydata()
    pts = np.vstack(line1_data).T
    
    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    line1, = ax.plot(line1_data[:, [0]], line1_data[:, [1]])  # plot line1
    line2, = ax.plot([], [])
    ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    linebuilder = plt_utils.LineBuilder(line2, 0.1)
    plt.show()
    line2_data = line2.get_xydata()
    pts2 = np.vstack(line2_data).T
    timer = time.time()
    match_table = match(pts, pts2)
    print(f"Time Elapsed: {time.time() - timer} s")
    print(f"Cost: {match_table[1, pts2.shape[1] - 1]}")
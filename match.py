import numpy as np

import bookstein
import pyshapetree

# A, B are 2 x N matrix
def match(A, B):

    # depth-first post-order
    def algorithm(sub_root_A, depth):
        B_num_pts = B.shape[1]
        table = np.zeros((B_num_pts, B_num_pts))

        # base case; no node, cost is zero
        if sub_root_A is None:
            return table
            
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
    pts = np.array([[-5.16129032, -5.29032258, -5.48387097, -5.51612903, -5.38709677, -4.32258065,
  -2.32258065, -1.29032258, -0.61290323, -0.12903226,  0.12903226,  0.16129032,
   0.19354839, -1.09677419, -3.09677419, -3.51612903, -3.58064516, -3.5483871,
  -3.,         -0.22580645,  0.90322581,  1.19354839,  1.12903226, -0.35483871,
  -1.96774194, -2.77419355, -2.96774194],
 [ 5.19480519,  5.28138528,  5.36796537,  5.41125541,  5.36796537,  5.06493506,
   4.71861472,  4.58874459,  4.24242424,  3.93939394,  3.76623377,  3.67965368,
   3.59307359,  1.6017316,   0.12987013,  0.04329004,  0.04329004,  0.,
  -0.12987013, -1.03896104, -1.9047619,  -2.29437229, -2.51082251, -4.37229437,
  -5.75757576, -6.14718615, -6.14718615]])
    pts2 = np.array([[-1.06451613, -0.93548387, -0.51612903,  0.4516129,   1.51612903,  2.35483871,
   2.61290323,  2.70967742,  2.77419355,  2.77419355,  2.35483871,  1.80645161,
   1.19354839,  0.74193548,  0.4516129,   0.38709677,  0.32258065,  0.35483871,
   1.09677419,  2.90322581,  4.03225806,  4.51612903,  4.70967742,  4.74193548,
   4.51612903,  2.5483871,   0.32258065, -0.83870968, -1.19354839],
 [ 7.05627706,  7.05627706,  7.05627706,  7.14285714,  6.92640693,  6.40692641,
   6.14718615,  5.97402597,  5.88744589,  5.71428571,  4.58874459,  3.59307359,
   2.5974026,   2.33766234,  2.16450216,  2.12121212,  2.03463203,  1.99134199,
   1.9047619,   1.34199134,  0.73593074,  0.17316017, -0.25974026, -0.47619048,
  -0.90909091, -2.51082251, -3.76623377, -4.24242424, -4.24242424]])
    import time
    timer = time.time()
    match_table = match(pts, pts)
    print(f"Time Elapsed: {time.time() - timer} s")
    #print(match_table)
    #print(match_table.shape)
    # Hmm, I would like cost to be 0
    print(f"Cost: {match_table[1, pts.shape[1] - 1]}")
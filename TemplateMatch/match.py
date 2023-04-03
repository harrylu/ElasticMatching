import numpy as np

# e is 3 x 1 column vector of (x, y, theta)
# weights is 3-tuple (alpha, beta, gamma) that weights x, y, theta distance
# paper recommends large theta weight
def event_distance(e_A, e_B, e_A1, e_B1, weights):
    x_A, y_A, theta_A = e_A
    x_B, y_B, theta_B = e_B
    x_A1, y_A1, _ = e_A1
    x_B1, y_B1, _ = e_B1

    dx = np.abs( (x_A - x_A1) - (x_B - x_B1))
    dy = np.abs( (y_A - y_A1) - (y_B - y_B1))
    dtheta = min(np.abs(theta_A - theta_B), 360 - np.abs(theta_A - theta_B))

    alpha, beta, gamma = weights
    devent = alpha * dx + beta * dy + gamma * dtheta
    return devent


# A, B are 3 x N matrices
# skip A = missing penalty; skip B = spurious penalty
# paper sets missing penalty = spurious penalty, so we do the same
# we only want 1 stroke, so we dont use stroke count difference penalty
def match(A, B, penalty, weights):
    # D is dp array
    D = np.empty((A.shape[1], B.shape[1]))
    D[0, 0] = 0
    for i in range(1, D.shape[0]):
        D[i, 0] = D[i - 1, 0] + penalty
    for j in range(1, D.shape[1]):
        D[0, j] = D[0, j - 1] + penalty
    e_A1, e_B1 = A[:, [1]], B[:, [1]]
    for i in range(1, D.shape[0]):
        for j in range(1, D.shape[1]):
            e_A, e_B = A[:, [i]], B[:, [j]]
            match_events = D[i - 1, j - 1] + event_distance(e_A, e_B, e_A1, e_B1, weights)
            skip_A = D[i - 1, j] + penalty
            skip_B = D[i, j - 1] + penalty
            skip_both = D[i - 1, j - 1] + 2 * penalty
            D[i, j] = min(match_events, skip_A, skip_B, skip_both)
    dist = D[D.shape[0] - 1, D.shape[1] - 1]
    final_dist = dist**2 / (penalty * A.shape[1] + penalty * B.shape[1])
    return final_dist

if __name__ == "__main__":
    a_str = """(0, 0, 223), (!2,!1, 223), (!1, 2, 223), (0, 5, 182), (1, 9, 194), (3, 13, 193), (5, 16, 190), (7, 19, 201), (10, 22, 239), (14, 24, 289), (16, 21, 261),
(16, 17, 196), (16, 13, 180), (16, 9, 188), (16, 5, 189), (15, 1, 182), (15,!2, 187), (14,!5, 178), (13,!9, 172), (13,!13, 187), (12,!17, 193),
(11,!21, 189), (9,!25, 183), (8,!28, 186), (6,!32, 196), (4,!35, 190), (1,!38, 185), (!1,!41, 207), (!4,!43, 211),
(!8,!44, 218), (!12,!45, 254), (!15,!42, 255), (!16,!38, 240), (!15,!34, 237), (!12,!31, 232), (!9,!30, 220),
(!5,!30, 211), (!1,!31, 196), (2,!33, 172), (6,!34, 172), (9,!35, 179), (13,!36, 172), (17,!37, 159), (21,!37, 155),
(25,!36, 155), (27,!35, 155)""".replace("!", "-")
    a = None
    exec(f"a = [{a_str}]")

    b_str = """(0, 0, 192), (2, 2, 192), (6, 3, 192), (10, 4, 187), (14, 5, 201), (18, 5, 216), (22, 4, 232), (25, 1, 251), (25,!1, 232), (24,!5, 200), (22,!9, 199),
(20,!12, 195), (17,!15, 191), (14,!18, 186), (11,!20, 180), (8,!23, 186), (5,!25, 264), (2,!27, 343), (3,!23, 286), (7,!21, 207),
(10,!19, 195), (14,!18, 203), (18,!17, 208), (22,!18, 203), (26,!19, 205), (29,!21, 211), (32,!24, 201), (34,!28, 192),
(36,!31, 202), (37,!35, 212), (37,!39, 203), (36,!43, 196), (35,!47, 195), (33,!50, 186), (31,!54, 195), (29,!57, 205),
(26,!59, 196), (22,!62, 200), (19,!63, 202), (15,!64, 202), (11,!64, 201), (7,!63, 196), (3,!62, 199), (0,!60, 213),
(!3,!58, 253), (!4,!54, 259), (!1,!51, 221), (1,!49, 199), (5,!47, 199), (5,!47, 199)""".replace("!", "-")
    b = None
    exec(f"b = [{b_str}]")

    c_str = """(0, 0, 61), (!3,!1, 61), (0, 0, 61), (3, 3, 193), (7, 4, 186), (11, 6, 185), (14, 7, 189), (18, 8, 188), (22, 9, 209), (26, 9, 243), (29, 7, 255), (30, 3, 230),
(29, 0, 204), (28,!4, 204), (26,!7, 196), (23,!10, 180), (21,!13, 191), (18,!16, 195), (15,!18, 184), (11,!21, 189), (8,!23, 260),
(5,!24, 330), (6,!21, 276), (9,!18, 207), (12,!16, 199), (16,!14, 209), (20,!13, 215), (24,!14, 213), (27,!15, 210),
(30,!18, 197), (33,!21, 201), (35,!24, 205), (36,!28, 196), (37,!32, 203), (37,!36, 203), (36,!40, 196), (35,!44, 195),
(33,!47, 192), (31,!51, 201), (28,!54, 201), (25,!56, 196), (22,!58, 199), (18,!59, 196), (14,!60, 201), (10,!60, 201),
(6,!59, 188), (2,!58, 199), (!1,!57, 204), (!4,!54, 192), (!7,!52, 201), (!9,!49, 210), (!11,!45, 234),
(!11,!41, 240), (!8,!38, 240), (!6,!36, 240)""".replace("!", "-")
    c = None
    exec(f"c = [{c_str}]")

    A = np.array(a).T
    B = np.array(b).T
    C = np.array(c).T

    weights = (.1, .1, 1)

    dist = match(A, B, 100, weights)
    print(dist)

    dist = match(B, C, 100, weights)
    print(dist)

    # man, we do have to do this empirically
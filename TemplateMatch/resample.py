import numpy as np

# pts is 2 x N np array
def resample(pts, distance):
    resampled = []
    dist_to_next_sample = distance
    for i in range(1, pts.shape[1]):
        prev_pt = pts[:, [i - 1]]
        curr_pt = pts[:, [i]]
        delta_dist = np.linalg.norm((curr_pt - prev_pt))
        if delta_dist < dist_to_next_sample:
            dist_to_next_sample -= delta_dist
            continue

        ratio = dist_to_next_sample/delta_dist
        sample_pt = (1 - ratio) * prev_pt + ratio * curr_pt
        resampled.append(sample_pt)

        sample_to_curr = curr_pt - sample_pt
        sample_to_curr_dist = np.linalg.norm(sample_to_curr)
        next_sample_offset = (distance / sample_to_curr_dist) * sample_to_curr
        
        num_samples = int(sample_to_curr_dist / distance)
        for _ in range(num_samples):
            resampled.append(resampled[-1] + next_sample_offset)
        leftover_dist = sample_to_curr_dist - distance * num_samples
        dist_to_next_sample = distance - leftover_dist
    return np.hstack(resampled)
         
        
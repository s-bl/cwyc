import numpy as np

def weighted_surprise(surprises, surprise_hist_weighting, hist_len):
    # Find last surprise signal and weight according to how long it is in the past
    surprises_idxs = [np.nonzero(surprises[:,i])[0] for i in range(surprises.shape[1])]
    surprises = np.asarray([0 if len(x) == 0 else surprise_hist_weighting**(hist_len-1-x[-1]) for x in surprises_idxs])

    return surprises

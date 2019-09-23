def mg_fn(x, mg):
    if x.ndim == 1:
        return x[mg]
    else:
        return x[:,mg]

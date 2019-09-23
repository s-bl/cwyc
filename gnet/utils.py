import matplotlib.pyplot as plt
import numpy as np

def plot_dense_weights(params, bias=None, xtiks=False):
    labels = []
    fig, ax = plt.subplots(1,3, figsize=(16,8))
    h,w = params.shape

    vmax = np.max([np.max(abs(params)), np.max(abs(bias))])
    vmin = -vmax

    # a_ij
    W_dense1 = np.zeros((h,h))
    k = 0
    for i in range(h):
        for j in range(i+1,h):
            W_dense1[i,j] = (params)[i,k]
            k += 1
    ax[0].matshow(W_dense1, cmap='bwr', vmin=vmin, vmax=vmax)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].xaxis.set_tick_params(rotation=90)
    ax[0].set_yticks(range(len(labels)))
    ax[0].set_yticklabels(labels)

    if not xtiks:
        ax[0].set_xticklabels(['']*(len(labels)))

    # b_ij
    W_dense2 = np.zeros((h,h))
    k = 0
    for i in range(h):
        for j in range(i+1,h):
            W_dense2[i,j] = (params)[j,k]
            k += 1
    ax[1].matshow(W_dense2, cmap='bwr', vmin=vmin, vmax=vmax)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels)
    ax[1].xaxis.set_tick_params(rotation=90)
    ax[1].set_yticks(range(len(labels)))
    ax[1].set_yticklabels(['']*(len(labels)))

    if not xtiks:
        ax[1].set_xticklabels(['']*(len(labels)))

    if bias is not None:
        W_dense3 = np.zeros((h,h))
        k = 0
        for i in range(h):
            for j in range(i+1,h):
                W_dense3[i,j] = bias[k]
                k += 1
        im = ax[2].matshow(W_dense3, cmap='bwr', vmin=vmin, vmax=vmax)
        ax[2].set_xticks(range(len(labels)))
        ax[2].set_xticklabels(labels)
        ax[2].xaxis.set_tick_params(rotation=90)
        ax[2].set_yticks(range(len(labels)))
        ax[2].set_yticklabels(['']*(len(labels)))

    fig.tight_layout()

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)

    #     fig.tight_layout()

    return fig

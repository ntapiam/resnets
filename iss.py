import torch
import numpy as np


def compute(x):
    print(f"Computing level 2 matrices on device: {x.device}")
    S = torch.zeros(2, x.shape[1], x.shape[1], device=x.device)
    dx = x.diff(axis=0)

    S = torch.vstack([S, torch.einsum('ti,tj->tij', x[1:-1] - x[0], dx[1:]).cumsum(axis=0)])

    return S

def pvar(n, p, d):
    running_pvar = np.zeros(n)

    for k in range(1, n):
        running_pvar[k] = np.max([running_pvar[j] + np.power(d(j, k), p) for j in range(k)])

    return np.power(running_pvar[-1], 1 / p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time, sys
    from tqdm import trange

    N = 100
    d = 64 * 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(N, d, device=DEVICE).cumsum(axis=0) / np.sqrt(N) #generate d-dimensional brownian path

    t0 = time.perf_counter()
    S = compute(x)
    t1 = time.perf_counter()
    print(f"Took {t1-t0:.>3f}s. Allocated {(sys.getsizeof(x.storage()) + sys.getsizeof(S.storage()))/ (1024 ** 2):.3f} MiB.")
    x, S = x.cpu(), S.cpu()

    def level1_dist(j, k):
        return torch.linalg.norm(x[k] - x[j])

    def level2_dist(j, k):
        return torch.linalg.norm(S[k] - S[j] + torch.outer(x[j], x[j]), ord='fro')

    ps = np.linspace(1, 3, 20)
    pvars = np.zeros(len(ps))
    print("Computing p-variations for p ∈ [1, 3]")
    for k in trange(len(ps)):
        pvars[k] = pvar(N, ps[k], level1_dist)
        if ps[k] > 2:
            pvars[k] += np.power(pvar(N, ps[k]/2, level2_dist), 0.5)

    plt.plot(ps, pvars)
    plt.xlabel('p')
    plt.show()

    

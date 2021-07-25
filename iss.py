import torch
import numpy as np


def compute(x):
    print(f"Computing level 2 matrices on device: {x.device}")
    S = torch.zeros(2, x.shape[1], x.shape[1], device=x.device)
    dx = x.diff(axis=0)

    S = torch.vstack([S, torch.einsum('ti,tj->tij', x[1:-1] - x[0], dx[1:]).cumsum(axis=0)])

    return S

def pvar(n, p, d, device='cpu'):
    running_pvar = torch.zeros(n, device=device)

    for k in range(1, n):
        dists = running_pvar[:k] + torch.pow(torch.tensor([d(j, k) for j in range(k)], device=device), p)
        running_pvar[k] = torch.max(dists)

    return torch.pow(running_pvar[-1], 1 / p)


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

    def level1_dist(j, k):
        return torch.linalg.norm(x[k] - x[j])

    def level2_dist(j, k):
        return torch.linalg.norm(S[k] - S[j] + torch.outer(x[j], x[j]), ord='fro')

    ps = np.linspace(1, 3, 20)
    pvars = np.zeros(len(ps))
    print("Computing p-variations for p ∈ [1, 3]")
    for k in trange(len(ps)):
        pvars[k] = pvar(N, ps[k], level1_dist, device=DEVICE)
        if ps[k] > 2:
            pvars[k] += torch.pow(pvar(N, ps[k]/2, level2_dist, device=DEVICE), 0.5)

    torch.save({'ps': ps, 'pvars': pvars}, 'pvar_data.pt')
    

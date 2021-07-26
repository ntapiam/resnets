import torch
import numpy as np


def compute(x):
    print(f"Computing level 2 matrices on device: {x.device}")
    S = torch.zeros(2, x.shape[1], x.shape[1], device=x.device)
    dx = x.diff(axis=0)

    S = torch.vstack(
        [S, torch.einsum("ti,tj->tij", x[1:-1] - x[0], dx[1:]).cumsum(axis=0)]
    )

    return S


def pvar(n, p, d, device="cpu"):
    running_pvar = torch.zeros(n, device=device)

    for k in range(1, n):
        dists = running_pvar[:k] + torch.pow(
            torch.tensor([d(j, k) for j in range(k)], device=device), p
        )
        running_pvar[k] = torch.max(dists)

    return torch.pow(running_pvar[-1], 1 / p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time, sys
    from tqdm import trange

    N = 100
    d = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(N, d, device=DEVICE).cumsum(axis=0) / np.sqrt(
        N
    )  # generate d-dimensional brownian path

    t0 = time.perf_counter()
    S = compute(x)
    t1 = time.perf_counter()
    print(
        f"Took {t1-t0:.>3f}s. Allocated {(sys.getsizeof(x.storage()) + sys.getsizeof(S.storage()))/(1024 ** 3):.3f} GiB."
    )

    pairwise_x = torch.linalg.vector_norm(x.unsqueeze(0) - x.unsqueeze(1), dim=-1)
    diag = torch.einsum("ti,tj->tij", x - x[0], x - x[0])
    pairwise_S = torch.pow(
        torch.linalg.matrix_norm(
            S.unsqueeze(0) - S.unsqueeze(1) + diag.unsqueeze(1), ord="fro"
        ),
        0.5,
    )

    def level1_dist(j, k):
        return pairwise_x[j, k]

    def level2_dist(j, k):
        return level1_dist(j, k) + pairwise_S[j, k]

    ps = np.linspace(1, 3, 20)
    pvars = np.zeros(len(ps))
    x, S = x.cpu(), S.cpu()
    total_memory = (
        sys.getsizeof(pairwise_x.storage())
        + sys.getsizeof(diag.storage())
        + sys.getsizeof(pairwise_S.storage())
    )
    total_memory /= 1024 ** 3
    print(
        "Computing p-variations for p ∈ [1, 3]. Allocated {} GiB".format(total_memory)
    )
    for k in trange(len(ps)):
        dist = level1_dist if ps[k] <= 2 else level2_dist
        pvars[k] = pvar(N, ps[k], dist)

    plt.plot(ps, pvars)
    plt.xlabel("p")
    plt.show()

    torch.save({"ps": ps, "pvars": pvars}, "pvar_data.pt")

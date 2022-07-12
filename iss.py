import torch


def compute(x):
    print(
        f"Computing level 2 matrices on device: {x.device}. Dimensions: (N={x.shape[0]}, d={x.shape[1]})"
    )
    S = torch.zeros(2, x.shape[1], x.shape[1], device=x.device)
    dx = x.diff(axis=0)

    S = torch.vstack(
        [S, ((x[1:-1] - x[0]).unsqueeze(1) * dx[1:].unsqueeze(-1)).cumsum(axis=0)]
    )

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time, sys
    from tqdm import trange
    import itertools as it
    import numpy as np
    from p_var import p_var
    from resnet import *

    n_layers = 128
    epochs = 100

    model = torch.load(f"resnet{n_layers}_relu_lin_e{epochs}.pth")
    x = torch.zeros(n_layers, 64 * 64)
    for (k, v) in enumerate(it.islice(model.parameters(), 8, 136)):
        x[k] = v.flatten()

    t0 = time.perf_counter()
    S = compute(x)
    t1 = time.perf_counter()
    memory = (sys.getsizeof(x.storage()) + sys.getsizeof(S.storage())) / (1024 ** 3)
    print(f"Took {t1-t0:.>3f}s. Allocated {memory:.3f} GiB.")

    # print("Precomputing pairwise norms.")
    # t0 = time.perf_counter()
    # pairwise_x = torch.linalg.vector_norm(x.unsqueeze(0) - x.unsqueeze(1), dim=-1)
    # xout = (x - x[0]).unsqueeze(1) * (x - x[0]).unsqueeze(-1)
    # Sjk = torch.zeros(N, N)

    # for j in range(N):
    #     for k in range(N):
    #         Sjk[j, k] = torch.pow(
    #             torch.linalg.matrix_norm(S[k] - S[j] + xout[j], ord="fro"), 0.5
    #         ).item()

    # t1 = time.perf_counter()
    # memory2 = (sys.getsizeof(pairwise_x.storage()) + sys.getsizeof(Sjk.storage())) / (
    #     1024 ** 3
    # )
    # print(
    #     f"Took {t1-t0:.>3f}s. Allocated {memory2:.>3f} GiB ({memory + memory2:.3f} GiB total)."
    # )

    # def level1_dist(j, k):
    #     return pairwise_x[j, k]

    # def level2_dist(j, k):
    #     return pairwise_x[j, k] + Sjk[j, k]

    ps = np.linspace(1, 3, 20)
    pvars = np.zeros(len(ps))
    x, S = x.cpu().detach().numpy(), S.cpu().reshape(S.shape[0], -1).detach().numpy()
    print("Computing p-variations for p ∈ [1, 3]")
    t0 = time.perf_counter()
    for k in trange(len(ps)):
        curr_pvar = p_var(x, ps[k], 'euclidean') ** (1 / ps[k])
        if ps[k] >= 2:
            curr_pvar += p_var(S, ps[k] / 2, 'euclidean') ** (2 / ps[k])
    t1 = time.perf_counter()
    print(f"Took {t1-t0:.>3f}s.")

    sns.set_theme()
    plt.plot(ps, pvars)
    plt.xlabel("p")
    plt.show()
    plt.savefig('pvar_plot.pdf')

    torch.save({"ps": ps, "pvars": pvars}, "pvar_data.pt")

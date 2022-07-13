import torch
from p_var import p_var


@torch.no_grad()
def compute(x):
    print(f"Computing level 2 matrices on device: {x.device}.\
        Dimensions: (N={x.shape[0]}, d={x.shape[1]})")
    S = torch.zeros(2, x.shape[1], x.shape[1], device=x.device)
    dx = x.diff(axis=0)

    S = torch.vstack([
        S,
        ((x[1:-1] - x[0]).unsqueeze(1) * dx[1:].unsqueeze(-1)).cumsum(axis=0)
    ])

    return S


if __name__ == "__main__":
    import itertools as it
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    from resnet import DenseBlock, ResNet, ResNetBlock

    n_layers = 512
    epochs = 100
    N = 64 * 64

    fname = Path(f"pvar_data_r{n_layers}_e{epochs}.pth")
    model_fname = Path(f"resnet{n_layers}_relu_lin_e{epochs}.pth")
    model = torch.load(model_fname)

    x = torch.zeros(n_layers, 64 * 64)
    for (k, v) in enumerate(it.islice(model.parameters(), 8, 8 + n_layers)):
        x[k] = v.flatten()

    if fname.is_file():
        print("Using saved data")
        data = torch.load(fname)
        pvars = data['pvars']
        ps = data['ps']
    else:
        import multiprocessing as mp
        import sys
        import time

        import numpy as np
        from tqdm import tqdm, trange

        with torch.no_grad():
            t0 = time.perf_counter()
            S = compute(x)
            t1 = time.perf_counter()
            memory = (sys.getsizeof(x.storage()) +
                      sys.getsizeof(S.storage())) / (1024**3)
            print(f"Took {t1-t0:.>3f}s. Allocated {memory:.3f} GiB.")

            print("Precomputing pairwise norms.")
            t0 = time.perf_counter()
            pairwise_x = (torch.linalg.vector_norm(x.unsqueeze(0) -
                                                   x.unsqueeze(1),
                                                   dim=-1).cpu().numpy())
            xout = (x - x[0]).unsqueeze(1) * (x - x[0]).unsqueeze(-1)
            Sjk = torch.zeros(N, N)

            for j in trange(n_layers):
                for k in trange(n_layers, leave=False):
                    Sjk[j, k] = torch.pow(
                        torch.linalg.matrix_norm(S[k] - S[j] + xout[j],
                                                 ord="fro"), 0.5).item()

            Sjk = Sjk.cpu().numpy()
            t1 = time.perf_counter()
            # memory2 = (sys.getsizeof(pairwise_x.storage()) + sys.getsizeof(Sjk.storage())) / (
            #     1024 ** 3
            # )
            # print(
            #     f"Took {t1-t0:.>3f}s. Allocated {memory2:.>3f} GiB ({memory + memory2:.3f} GiB total)."
            # )

            ps = np.linspace(1, 3, 20)
            print("Computing p-variations for p ∈ [1, 3]")
            t0 = time.perf_counter()
            with mp.Pool() as p:
                inputs = zip(it.repeat(n_layers), ps, it.repeat(pairwise_x))
                pvars = np.array(p.starmap(p_var,
                                           tqdm(inputs,
                                                total=len(ps))))**(1 / ps)
                inputs = zip(it.repeat(n_layers), ps[ps >= 2], it.repeat(Sjk))
                pvars[ps >= 2] += np.array(
                    p.starmap(p_var,
                              tqdm(inputs,
                                   total=len(ps[ps >= 2]))))**(1 / ps[ps >= 2])

            t1 = time.perf_counter()
            print(f"Took {t1-t0:.>3f}s.")
            torch.save({"ps": ps, "pvars": pvars}, fname)

    sns.set_theme()
    plt.plot(ps[ps < 2], pvars[ps < 2], label=r"$||\mathbf{w}||_p$")
    plt.plot(ps[ps >= 2], pvars[ps >= 2], label=r"$|||\mathbb{W}|||_p$")
    plt.xlabel(r"$p$")
    plt.legend()
    plt.savefig(f"pvar_plot_r{n_layers}_e{epochs}.pdf")

    with torch.no_grad():
        plt.clf()
        dW = x[:, 1].clone()
        W = dW.cumsum(dim=0) - dW[0]
        dW /= dW.abs().max()
        W /= W.abs().max()
        plt.plot(1 + torch.arange(n_layers),
                 dW,
                 label=r"$\mathbf{w}^1_{k+1}-\mathbf{w}^1_k$")
        plt.plot(1 + torch.arange(n_layers), W, label=r"$\mathbf{w}^1_k-\mathbf{w}^1_0$")
        plt.xlabel(r"layer ($k$)")
        plt.legend()
        plt.savefig(f"evol_plot_r{n_layers}_e{epochs}.pdf")

import matplotlib.pyplot as plt
import pandas as pd


def to_gflops(n, ms):
    ops = n ** 3
    flops = ops / (ms / 1e3)
    return flops / 1e9


def prepare_perf(ax):
    ax.set_xlabel("N")
    ax.set_ylabel("time[ms]")
    plt.yscale("log")
    ax.grid(axis="y", which="major", color="0.6", linestyle="-")
    ax.grid(axis="y", which="minor", color="0.8", linestyle="-")
    ax.set_axisbelow(True)


def prepare_gflops(ax):
    ax.set_xlabel("N")
    ax.set_ylabel("GFlops")
    plt.yscale("log")
    ax.grid(axis="y", which="major", color="0.6", linestyle="-")
    ax.grid(axis="y", which="minor", color="0.8", linestyle="-")
    ax.set_axisbelow(True)


# mpi: n - t graph (every t)
def plot_mpi_perf():
    df = pd.read_csv("./../mpi/result_pn.txt")
    grouped = df.groupby(["n", "p"])
    mn = pd.DataFrame(grouped.mean())
    mn = mn.unstack()
    ax = mn.plot.line(marker="o")
    prepare_perf(ax)
    ax.legend(tuple([f"processes = {p}" for (_, p) in mn.columns]))
    plt.savefig("./fig1.png")
    plt.clf()


def df_n_gflops(path: str):
    df = pd.read_csv(path)
    df = df.groupby(["n"]).mean().reset_index()
    df["gflops"] = df.apply(lambda r: to_gflops(r.n, r.time), axis=1)
    return df


# openmp, mpi, cuda: n - GFlops graph
def plot_gflops():
    df_omp = df_n_gflops("./../openmp/result.txt")
    df_mpi = df_n_gflops("../mpi/result_n.txt")
    df_cuda = df_n_gflops("../cuda/result.txt")
    ax = df_omp.plot.line(x="n", y="gflops", label="openmp", marker="o")
    ax = df_mpi.plot.line(x="n", y="gflops", ax=ax, label="mpi", marker="o")
    ax = df_cuda.plot.line(x="n", y="gflops", ax=ax, label="cuda", marker="o")
    prepare_gflops(ax)
    plt.savefig("./fig2.png")
    plt.clf()


if __name__ == "__main__":
    plot_mpi_perf()
    plot_gflops()
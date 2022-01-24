import matplotlib.pyplot as plt
import pandas as pd


def to_gflops(n, time):
    ops = n ** 3
    flops = ops / (time / 1e3)
    return flops / 1e9


font = "Ricty Diminished"
y = -0.3


def prepare(ax):
    ax.set_xlabel("N")
    ax.set_ylabel("GFlops")
    ax.grid(axis="y", which="major", color="0.6", linestyle="-")
    ax.grid(axis="y", which="minor", color="0.8", linestyle="-")
    ax.set_axisbelow(True)


def plot_openmp(path: str):
    df = pd.read_csv(path)
    grouped = df.groupby(["n", "t"])
    mn = pd.DataFrame(grouped.mean())
    mn = mn.apply(lambda r: to_gflops(r.name[0], r.time), axis=1)
    mn = mn.unstack()
    print(mn)
    ax = mn.plot.line(marker="o")
    prepare(ax)
    # plt.title(
    #     r"Open MPでスレッド数を変えた時の $N \times N$ 行列積の性能",
    #     fontname=font,
    #     y=y,
    # )
    # ax.legend(tuple([f"threads = {t}" for (_, t) in mn.columns]))
    # plt.show()
    plt.savefig("./result_openmp.png")
    plt.clf()


def plot_mpi(path: str):
    df = pd.read_csv(path)
    grouped = df.groupby(["n", "p"])
    mn = pd.DataFrame(grouped.mean())
    mn = mn.apply(lambda r: to_gflops(r.name[0], r.time), axis=1)
    mn = mn.unstack()
    print(mn)

    ax = mn.plot.line(marker="o")
    prepare(ax)
    # plt.title(
    #     r"MPIでプロセス数を変えた時の $N \times N$ 行列積の性能",
    #     fontname=font,
    #     y=y,
    # )
    # ax.legend(tuple([f"processes = {p}" for (_, p) in mn.columns]))
    # plt.show()
    plt.savefig("./result_mpi.png")
    plt.clf()


def plot_cuda(path: str):
    df = pd.read_csv(path)
    grouped = df.groupby(["n"])
    mn = pd.DataFrame(grouped.mean())
    mn = mn.apply(lambda r: to_gflops(r.name, r.time), axis=1)
    print(mn)

    ax = mn.plot.line(marker="o")
    # plt.title(
    #     r"1 $\times$ 1 block, 32 $\times$ 32 thread/block での $N \times N$ 行列積の性能",
    #     fontname=font,
    #     y=y,
    # )
    prepare(ax)
    # plt.show()
    plt.savefig("./result_cuda.png")
    plt.clf()


if __name__ == "__main__":
    plot_openmp("./../openmp/result.txt")
    plot_mpi("./../mpi/result.txt")
    plot_cuda("./../cuda/result.txt")

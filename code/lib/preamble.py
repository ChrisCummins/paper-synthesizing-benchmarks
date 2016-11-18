# preamble

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sys

from collections import Counter, defaultdict, OrderedDict
from functools import partial
from io import StringIO
from IPython.core.display import display, HTML
from itertools import permutations
from labm8 import fmt
from labm8 import fs
from labm8 import math as labmath
from labm8 import time
from labm8 import viz
from math import sqrt, ceil
from numpy.random import RandomState
from random import random
from random import seed
from shutil import move
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# code for the paper:
import clgen
import clgen.config
import clgen.preprocess
import clgen.sampler
import cgo13

from clgen import config as cfg

# plotting config:
sns.set(style="ticks", color_codes=True)
plt.style.use(["seaborn-white", "seaborn-paper"])

# tables config:
pd.set_option('display.max_rows', 15)

# warn if CLgen doesn't have OpenCL support
if not cfg.USE_OPENCL:
    print("warning: CLgen does not have OpenCL support. Some of the "
          "experiments in this notebook are disabled.", file=sys.stderr)


def has_opencl():
    """determine if platform supports cldrive"""
    return cfg.USE_OPENCL


def can_reproduce_experiments():
    """determine if platform can reproduce experiments"""
    if cfg.USE_OPENCL:
        import pyopencl as cl
        import clgen.cldrive
        try:
            clgen.cldrive.init_opencl(cl.device_type.CPU)
            clgen.cldrive.init_opencl(cl.device_type.GPU)
            return True
        except Exception:
            return False
    else:
        return False

# warn if CLgen version is incorrect
REQUIRED_CLGEN_VERSION = "0.1.7"
if clgen.version() != REQUIRED_CLGEN_VERSION:
    print("warning: This notebook requires CLgen version {required}. "
          "You have version {actual} installed."
          .format(required=REQUIRED_CLGEN_VERSION, actual=clgen.version()),
          file=sys.stderr)
    print("         There may be incompatabilities.", file=sys.stderr)


class DictTable(dict):
    """takes a dict and renders an HTML table"""
    def _repr_html_(self):
        html = ["<table width=100%>"]
        for key, value in self.items():
            html.append("<tr>")
            html.append("<td><b>{0}</b></td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


def line_word_char_count(path):
    """count words, lines, chars in file"""
    num_lines = 0
    num_words = 0
    num_chars = 0

    with open(path) as infile:
        for line in infile:
            words = line.split()

            num_lines += 1
            num_words += len(words)
            num_chars += len(line)

    return num_lines, num_words, num_chars


def rand_jitter(arr, factor=0.01, randomstate=RandomState(204)):
    """apply jitter to array"""
    stdev = factor * (max(arr) - min(arr))
    return arr + randomstate.randn(len(arr)) * stdev


def scatter_with_jitter(plt, x, y, **kwargs):
    """scatter x,y values with jitter"""
    jitter_opts = kwargs.get("jitter_opts", {})
    if "jitter_opts" in kwargs:
        kwargs.pop("jitter_opts")

    return plt.scatter(rand_jitter(x, **jitter_opts),
                       rand_jitter(y, **jitter_opts), **kwargs)


def shortlabels(groups):
    """shorten benchmark suite names"""
    return [escape_suite_name(re.sub("-.+$", "", x)) for x in groups]


def shortbenchmark(benchmark):
    """short benchmark name"""
    return benchmark.split('-')[-1]


def escape_benchmark_name(g):
    """escape benchmark name for display"""
    c = g.split('-')
    return escape_suite_name(g) + "." + c[-2]


def plot_pca(X, B_out, Bother=None, pca=None):
    """plot PCA projection of feature space"""
    def jitter_opts(randomstate):
        return {"factor": .075, "randomstate": RandomState(randomstate)}

    # size and opacity
    plot_opts = {"s": 85, "alpha": .65}

    # apply jitter and repack
    x, y = zip(*X)
    x = rand_jitter(x, **jitter_opts(204))
    y = rand_jitter(y, **jitter_opts(205))
    X = list(zip(x, y))

    # group by correct or not
    correct   = [x for x, b in zip(X, B_out.to_dict('records')) if     b["p_correct"]]
    incorrect = [x for x, b in zip(X, B_out.to_dict('records')) if not b["p_correct"]]

    if Bother is not None:
        additional = pca.transform(get_raw_features(Bother))
        scatter_with_jitter(plt, *zip(*additional), color="g", marker="o",
                            label="Additional", jitter_opts=jitter_opts(206),
                            **plot_opts)
    plt.scatter(*zip(*incorrect),
                color="r", marker="v", label='Incorrect', **plot_opts)
    plt.scatter(*zip(*correct),
                color="b", marker="^", label='Correct', **plot_opts)

    # no tick labels
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # axis labels
    plt.xlabel(r"Principle Component 1 $\rightarrow$", ha="right")
    plt.ylabel(r"Principle Component 2 $\rightarrow$", ha="right")

    # position axis labels at end of axis
    ax.xaxis.set_label_coords(1, -.025)
    ax.yaxis.set_label_coords(-.025, 1)

    # show legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.get_legend().draw_frame(True)

    return ax


def get_our_model():
    """return extended model"""
    return KNeighborsClassifier(1)


def get_our_features(D):
    """return extended featureset"""
    return np.array([
        D["comp"].values,
        D["rational"].values,
        D["mem"].values,
        D["localmem"].values,
        D["coalesced"].values,
        D["transfer"].values,
        D["wgsize"].values,
        (D["transfer"].values / (D["comp"].values + D["mem"].values)),
        (D["coalesced"].values / D["mem"].values),
        ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),
        (D["comp"].values / D["mem"].values),
    ]).T


def get_raw_features(D):
    """return raw feature values"""
    return np.array([
        D["comp"].values,
        D["rational"].values,
        D["mem"].values,
        D["localmem"].values,
        D["coalesced"].values,
        D["atomic"].values,
        D["transfer"].values,
        D["wgsize"].values,
    ]).T


def get_cgo13_features(D):
    """return features used in CGO'13"""
    return np.array([
        (D["transfer"].values / (D["comp"].values + D["mem"].values)),
        (D["coalesced"].values / D["mem"].values),
        ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),
        (D["comp"].values / D["mem"].values),
    ]).T


def readfile(path):
    """read file to string"""
    with open(path) as infile:
        return ''.join(infile.readlines())


def escape_suite_name(g):
    """format benchmark suite name for display"""
    c = g.split('-')
    if (c[0] == "amd" or c[0] == "npb" or c[0] == "nvidia" or c[0] == "shoc"):
        return c[0].upper()
    else:
        return c[0].capitalize()


def get_nearest_neighbour_distance(F1, F2):
    """return nearest-neighbour distances from F1 to F2"""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(F2)
    distances, indices = nbrs.kneighbors(F1)
    return distances


def complete(condition=True, msg=None):
    """passed/failed display"""
    if msg is None:
        msg_html = ""
    else:
        msg_html = """
        <div style="margin-top:-.7em; padding-bottom:.2em;">{msg}</div>
        """.format(msg=msg)

    if condition:
        html = """
<div style="background-color:#5cb85c; color:#fff; text-align:center; border-radius:10px;">
  <h1 style="padding:.5em; font-weight:400;">&#9745; Complete</h1>
  {message}
</div>
""".format(message=msg_html)
    else:
        html = """
<div style="background-color:#d9534f; color:#fff; text-align:center; border-radius:10px;">
  <h1 style="padding:.5em; font-weight:400;">&#9746; Failed</h1>
  {message}
</div>
""".format(message=msg_html)

    display(HTML(html))


def header(*msg, sep=" "):
    """html header"""
    display(HTML("<h3>{msg}</h3>".format(msg=sep.join(msg))))


def plot_speedups_with_clgen(benchmarks_data, clgen_data, suite="npb"):
    """
    Plot speedups of predictive models trained with and without clgen.

    Returns speedups (without and with).
    """
    # datasets: B - benchmarks, S - synthetics, BS - benchmarks + synthetics:
    B = pd.read_csv(benchmarks_data)
    B["group"] = ["B"] * len(B)

    S = pd.read_csv(clgen_data)
    S["group"] = ["S"] * len(S)

    BS = pd.concat((B, S))

    # find the ZeroR. This is the device which is most frequently optimal
    Bmask = B[B["benchmark"].str.contains(suite)]
    zeror = Counter(Bmask["oracle"]).most_common(1)[0][0]
    zeror_runtime = "runtime_" + zeror.lower()

    # get the names of the benchmarks, in the form: $suite-$version-$benchmark
    benchmark_names = sorted(set([
        re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)-", b).group(1)
        for b in B["benchmark"] if b.startswith(suite)
    ]))

    B_out, BS_out = [], []
    for benchmark in benchmark_names:
        clf = cgo13.model()
        features = get_cgo13_features
        # cross validate on baseline
        B_out += cgo13.leave_one_benchmark_out(clf, features, B, benchmark)
        # reset model
        clf = cgo13.model()
        # repeate cross-validation with synthetic kernels
        BS_out += cgo13.leave_one_benchmark_out(clf, features, BS, benchmark)

    # create results frame
    R_out = []
    for b, bs in zip(B_out, BS_out):
        # get runtimes of device using predicted device
        b_p_runtime = b["runtime_" + b["p"].lower()]
        bs_p_runtime = bs["runtime_" + bs["p"].lower()]

        # speedup is the ratio of runtime using the predicted device
        # over runtime using ZeroR device
        b["p_speedup"] = b_p_runtime / b[zeror_runtime]
        bs["p_speedup"] = bs_p_runtime / bs[zeror_runtime]

        if "training" in benchmarks_data:
            # $benchmark
            group = escape_benchmark_name(b["benchmark"])
        else:
            # $benchmark.$dataset
            group = re.sub(r"[^-]+-[0-9\.]+-([^-]+)-.+", r"\1",
                           b["benchmark"]) + "." + b["dataset"]
        b["group"] = group
        bs["group"] = group

        # set the training data type
        b["training"] = "Grewe et al."
        bs["training"] = "w. CLgen"

        R_out.append(b)
        R_out.append(bs)

    R = pd.DataFrame(R_out)

    b_mask = R["training"] == "Grewe et al."
    bs_mask = R["training"] == "w. CLgen"

    B_speedup = labmath.mean(R[b_mask].groupby(["group"])["p_speedup"].mean())
    BS_speedup = labmath.mean(R[bs_mask].groupby(["group"])["p_speedup"].mean())

    print("  #. benchmarks:                  ",
          len(set(B["benchmark"])), "kernels,", len(B), "observations")
    print("  #. synthetic:                   ",
          len(set(S["benchmark"])), "kernels,", len(S), "observations")
    print()
    print("  ZeroR device:                    {}".format(zeror))
    print()
    print("  Speedup of Grewe et al.:         {:.2f} x".format(B_speedup))
    print("  Speedup w. CLgen:                {:.2f} x".format(BS_speedup))

    R = R.append({  # average bars
        "group": "Average",
        "p_speedup": B_speedup,
        "training": "Grewe et al."
    }, ignore_index=True)
    R = R.append({
        "group": "Average",
        "p_speedup": BS_speedup,
        "training": "w. CLgen"
    }, ignore_index=True)

    R["p_speedup"] -= 1  # negative offset so that bars start at 1

    # colors
    palette = sns.cubehelix_palette(len(set(R["training"])),
                                    rot=-.4, light=.85, dark=.35)

    ax = sns.barplot(
        x="group", y="p_speedup", data=R, ci=None, hue="training",
        palette=palette)
    plt.ylabel("Speedup")
    plt.xlabel("")

    plt.axhline(y=0, color="k", lw=1)  # speedup line
    plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1, linestyle="--")  # average line

    ax.get_legend().set_title("")  # no legend title
    plt.legend(loc='upper right')
    ax.get_legend().draw_frame(True)

    # plot shape and size
    figsize = (9, 2.2)
    if "nvidia" in benchmarks_data:
        typecast = int; plt.ylim(-1, 16)
    elif "training" in benchmarks_data:
        typecast = float; figsize = (7, 3.2)
    else:
        typecast = float

    # counter negative offset:
    ax.set_yticklabels([typecast(i) + 1 for i in ax.get_yticks()])

    plt.setp(ax.get_xticklabels(), rotation=90)

    viz.finalise(figsize=figsize, tight=True)
    return B_speedup, BS_speedup


def _compare_clfs(clf1, get_features1, clf2, get_features2, D1, D2, benchmark):
    """cross-validate across all benchmarks using CGO13 model and our own, with
    and without synthetic benchmarks. Report per-platform speedup of our model
    over CGO13"""
    test1_mask = D1["benchmark"].str.contains(r"^" + benchmark)
    test2_mask = D2["benchmark"].str.contains(r"^" + benchmark)
    assert(len(D1[test1_mask]) == len(D2[test2_mask]))

    # create data masks. For training we exclude all results from benchmark
    train1_mask = ~test1_mask
    train2_mask = ~test2_mask

    # create training and testing data
    X1_train = get_features1(D1.loc[train1_mask])
    X2_train = get_features2(D2.loc[train2_mask])
    y1_train = cgo13.getlabels(D1[train1_mask])
    y2_train = cgo13.getlabels(D2[train2_mask])

    D1_test = D1[test1_mask]
    D2_test = D2[test2_mask]
    X1_test = get_features1(D1.loc[test1_mask])
    X2_test = get_features2(D2.loc[test2_mask])
    y1_test = cgo13.getlabels(D1_test)
    y2_test = cgo13.getlabels(D2_test)

    clf1.fit(X1_train, y1_train)  # train classifiers
    clf2.fit(X2_train, y2_train)

    predicted1 = clf1.predict(X1_test)  # make predictions
    predicted2 = clf2.predict(X2_test)

    D_out = []
    for d, y, p1, p2 in zip(D1_test.to_dict('records'), y1_test,
                            predicted1, predicted2):
        d["p1"], d["p2"] = p1, p2
        D_out.append(d)

    return D_out  # return a list of dicts


def plot_speedups_extended_model_2platform(platform_a, platform_b):
    """
    Plot speedup of extended model over Grewe et al for 2 platforms
    """
    aB = pd.read_csv(platform_a[0])
    aB["synthetic"] = np.zeros(len(aB))
    bB = pd.read_csv(platform_b[0])
    bB["synthetic"] = np.zeros(len(bB))
    B = pd.concat((aB, bB))

    aS = pd.read_csv(platform_a[1])
    aS["synthetic"] = np.ones(len(aS))
    bS = pd.read_csv(platform_b[1])
    bS["synthetic"] = np.ones(len(bS))
    S = pd.concat((aS, bS))

    aBS = pd.concat((aB, aS))
    bBS = pd.concat((bB, bS))
    BS = pd.concat((B, S))

    assert(len(B) == len(aB) + len(bB))  # sanity checks
    assert(len(S) == len(aS) + len(bS))
    assert(len(BS) == len(aBS) + len(bBS))

    # get benchmark names: <suite>-<benchmark>
    benchmark_names = sorted(set([
        re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
        for b in B["benchmark"]
    ]))

    # perform cross-validation
    B_out = []
    for i, benchmark in enumerate(benchmark_names):
        print("\ranalyzing", i + 1, benchmark, end="")
        cgo13_clf, our_clf = cgo13.model(), get_our_model()
        cgo13_features, our_features = get_cgo13_features, get_our_features

        # cross validate on Grewe et al. and our model
        tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                            aBS, aBS, benchmark)
        for d in tmp: d["platform"] = "AMD Tahiti 7970"
        B_out += tmp

        # reset models
        cgo13_clf, our_clf = cgo13.model(), get_our_model()

        # same as before, on other platform:
        tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                            bBS, bBS, benchmark)
        for d in tmp: d["platform"] = "NVIDIA GTX 970"
        B_out += tmp
    print()

    # create results frame
    R_out = []
    # get runtimes of device using predicted device
    for b in B_out:
        p1_runtime = b["runtime_" + b["p1"].lower()]
        p2_runtime = b["runtime_" + b["p2"].lower()]

        # speedup is the ratio of runtime using our predicted device
        # over runtime using CGO13 predicted device.
        b["p_speedup"] = p2_runtime / p1_runtime

        # get the benchmark name
        b["group"] = escape_benchmark_name(b["benchmark"])

        R_out.append(b)
    R = pd.DataFrame(R_out)

    improved = R[R["p_speedup"] > 1]

    Amask = R["platform"] == "AMD Tahiti 7970"
    Bmask = R["platform"] == "NVIDIA GTX 970"
    a = R[Amask]
    b = R[Bmask]

    a_speedups = a.groupby(["group"])["p_speedup"].mean()
    b_speedups = b.groupby(["group"])["p_speedup"].mean()

    a_speedup = labmath.mean(a_speedups)
    b_speedup = labmath.mean(b_speedups)

    assert(len(R) == len(a) + len(b))  # sanity-check

    print("  #. benchmarks:          ",
          len(set(B["benchmark"])), "kernels,", len(B), "observations")
    print("  #. synthetic:           ",
          len(set(S["benchmark"])), "kernels,", len(S), "observations")
    print()
    print("  Speedup on AMD:          {:.2f} x".format(a_speedup))
    print("  Speedup on NVIDIA:       {:.2f} x".format(b_speedup))

    palette = sns.cubehelix_palette(
        len(set(R["platform"])), start=4, rot=.8, light=.8, dark=.3)

    R = R.append({  # average bars
        "group": "Average",
        "p_speedup": a_speedup,
        "platform": "AMD Tahiti 7970"
    }, ignore_index=True)
    R = R.append({
        "group": "Average",
        "p_speedup": b_speedup,
        "platform": "NVIDIA GTX 970"
    }, ignore_index=True)

    R["p_speedup"] -= 1  # negative offset so that bars start at 1

    ax = sns.barplot(x="group", y="p_speedup", hue="platform", data=R,
                     palette=palette, ci=None)

    plt.ylabel("Speedup over Grewe et al."); plt.xlabel("")

    plt.axhline(y=0, color="k", lw=1)
    plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1, linestyle="--")
    plt.ylim(-1, 9)
    plt.setp(ax.get_xticklabels(), rotation=90)  # rotate x ticks
    ax.get_legend().set_title("")  # legend
    plt.legend(loc='upper right')

    # counter negative offset
    ax.set_yticklabels([int(i) + 1 for i in ax.get_yticks()])

    ax.get_legend().draw_frame(True)

    viz.finalise(figsize=(9, 4), tight=True)


def plot_speedups_extended_model(benchmarks_data, clgen_data):
    """
    Plots speedups of extended model over Grewe et al

    Returns: speedup
    """
    B = pd.read_csv(benchmarks_data)
    B["synthetic"] = np.zeros(len(B))

    S = pd.read_csv(clgen_data)
    S["synthetic"] = np.ones(len(S))

    BS = pd.concat((B, S))

    assert(len(BS) == len(B) + len(S))

    # get benchmark names: <suite>-<benchmark>
    benchmark_names = sorted(set([
        re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
        for b in B["benchmark"]
    ]))

    # perform cross-validation
    B_out = []
    for i, benchmark in enumerate(benchmark_names):
        print("\ranalyzing", i + 1, benchmark, end="")
        cgo13_clf, our_clf = cgo13.model(), get_our_model()
        cgo13_features, our_features = get_cgo13_features, get_our_features

        # cross validate on Grewe et al. and our model
        tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                            BS, BS, benchmark)
        B_out += tmp
    print()

    # create results frame
    R_out = []
    # get runtimes of device using predicted device
    for b in B_out:
        p1_runtime = b["runtime_" + b["p1"].lower()]
        p2_runtime = b["runtime_" + b["p2"].lower()]

        # speedup is the ratio of runtime using our predicted device
        # over runtime using CGO13 predicted device.
        b["p_speedup"] = p2_runtime / p1_runtime

        # get the benchmark name
        b["group"] = escape_benchmark_name(b["benchmark"])

        R_out.append(b)
    R = pd.DataFrame(R_out)

    improved = R[R["p_speedup"] > 1]

    speedups = R.groupby(["group"])["p_speedup"].mean()
    speedup = labmath.mean(speedups)

    print("  #. benchmarks:          ",
          len(set(B["benchmark"])), "kernels,", len(B), "observations")
    print("  #. synthetic:           ",
          len(set(S["benchmark"])), "kernels,", len(S), "observations")
    print()
    print("  Speedup:                 {:.2f} x".format(speedup))

    palette = sns.cubehelix_palette(1, start=4, rot=.8, light=.8, dark=.3)

    R = R.append({  # average bar
        "group": "Average",
        "p_speedup": speedup
    }, ignore_index=True)

    R["p_speedup"] -= 1  # negative offset so that bars start at 1

    ax = sns.barplot(x="group", y="p_speedup", data=R,
                     palette=palette, ci=None)

    plt.ylabel("Speedup over Grewe et al."); plt.xlabel("")

    plt.axhline(y=0, color="k", lw=1)
    plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1, linestyle="--")
    plt.ylim(-1, 9)
    plt.setp(ax.get_xticklabels(), rotation=90)  # rotate x ticks

    # counter negative offset
    ax.set_yticklabels([int(i) + 1 for i in ax.get_yticks()])

    viz.finalise(figsize=(7, 3.7), tight=True)
    return speedup

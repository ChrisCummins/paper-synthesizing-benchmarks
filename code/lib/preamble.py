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
from itertools import permutations
from labm8 import fmt
from labm8 import fs
from labm8 import math as labmath
from labm8 import viz
from math import sqrt, ceil
from numpy.random import RandomState
from random import random
from random import seed
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

# plotting config:
sns.set(style="ticks", color_codes=True)
plt.style.use(["seaborn-white", "seaborn-paper"])

# tables config:
pd.set_option('display.max_rows', 15)

# warn if CLgen doesn't have OpenCL support
from clgen import config as cfg
if not cfg.USE_OPENCL:
    print("warning: CLgen does not have OpenCL support. Some of the "
          "experiments in this notebook will fail.", file=sys.stderr)

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


def escape_benchmark_name(g):
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
   return KNeighborsClassifier(1)


def get_our_features(D):
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


def summarize_distance(distances):
    return

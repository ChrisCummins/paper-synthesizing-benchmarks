#
# cgo13 - Implementation of the autotuner from:
#
#     Grewe, D., Wang, Z., & O'Boyle, M. F. P. M. (2013). Portable
#     Mapping of Data Parallel Programs to OpenCL for Heterogeneous
#     Systems. In CGO. IEEE.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import re

from collections import defaultdict
from labm8 import flatten, fs, math as labmath
from sklearn import model_selection as cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def ingroup(getgroup, d, group):
    return getgroup(d) == group


def getsuite(d):
    return re.match(r"^[a-zA-Z-]+-[0-9\.]+", d["benchmark"]).group(0)


def getbenchmark(d):
    return re.sub(r"-[^-]+$", "", d["benchmark"])


def getprog(d):
    return re.match(r"^[a-zA-Z-]+-[0-9\.]+-[^-]+-", d["benchmark"]).group(0)


def getclass(d):
    return d["oracle"]


class DataFilter(object):
    @staticmethod
    def from_str(string):
        pass


class GetGroup(object):
    @staticmethod
    def from_str(string):
        pass


def _eigenvectors(data):
    F1 = data["F1_norm"]
    F2 = data["F2_norm"]
    F3 = data["F3_norm"]
    F4 = data["F4_norm"]

    # Eigenvectors: (hardcoded values as computed by Weka)
    V1 = - .1881 * F1 + .6796 * F2 - .2141 * F3 - .6760 * F4
    V2 = - .7282 * F1 - .0004 * F2 + .6852 * F3 + .0149 * F4
    V3 = - .6590 * F1 - .1867 * F2 - .6958 * F3 - .2161 * F4
    V4 =   .0063 * F1 + .7095 * F2 + .0224 * F3 - .7044 * F4

    return (V1, V2, V3, V4)


def normalize(array):
    factor = np.amax(array)
    return np.copy(array) / factor


class LabelledData(object):
    @staticmethod
    def from_csv(path, group_by=None):
        getgroup = {
            "class": getclass,
            "suite": getsuite,
            "prog": getprog,
            "benchmark": getbenchmark,
        }.get(group_by, lambda x: "None")

        data = pd.read_csv(path)

        data["Group"] = [getgroup(d) for d in data.to_dict(orient='records')]
        data["F1_norm"] = normalize(data["F1:transfer/(comp+mem)"])
        data["F2_norm"] = normalize(data["F2:coalesced/mem"])
        data["F3_norm"] = normalize(data["F3:(localmem/mem)*avgws"])
        data["F4_norm"] = normalize(data["F4:comp/mem"])

        return data


class UnLabelledData(object):
    @staticmethod
    def from_csv(path):
        data = pd.read_csv(smith.assert_exists(path),
                           names=["benchmark", "dataset", "kernel",
                                  "wgsize", "transfer", "runtime", "ci"])

        return data


def norm_feature_distance(f1, f2):
    """
    Distance between two features (as dicts).
    """
    d1 = abs(f1["F1_norm"] - f2["F1_norm"])
    d2 = abs(f1["F2_norm"] - f2["F2_norm"])
    d3 = abs(f1["F3_norm"] - f2["F3_norm"])
    d4 = abs(f1["F4_norm"] - f2["F4_norm"])

    return math.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4)


def eigens_distance(f1, f2):
    """
    Distance between two features (as dicts).
    """
    d1 = abs(f1["E1"] - f2["E1"])
    d2 = abs(f1["E2"] - f2["E2"])
    d3 = abs(f1["E3"] - f2["E3"])
    d4 = abs(f1["E4"] - f2["E4"])

    return math.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4)


def nearest_neighbours(data1, data2, same_class=False,
                       distance=norm_feature_distance):
    """
    Find the minimum distances between datapoints.

    Returns list of tuples, where each tuple is in the form:

       (distance, index_of_closest, same_oracle)
    """
    dists, indices, sameoracles = [], [], []

    for d1 in data1.to_dict(orient="record"):
        mindist, index, sameoracle = float('inf'), None, False
        for i, d2 in enumerate(data2.to_dict(orient="record")):
            if not d1 == d2:
                dist = distance(d1, d2)
                if ((not same_class) or
                    (same_class and d1["oracle"] == d2["oracle"])):
                    if dist < mindist and i not in indices:
                        mindist = dist
                        index = i
                        sameoracle = d1["oracle"] == d2["oracle"]
        dists.append(mindist)
        indices.append(index)
        sameoracles.append(sameoracle)
    return zip(dists, indices, sameoracles)


# feature extractors

def cgo13_features(d):
    return np.array([
        d["F1:transfer/(comp+mem)"],
        d["F2:coalesced/mem"],
        d["F3:(localmem/mem)*avgws"],
        d["F4:comp/mem"]
    ]).T


def raw_features(d):
    """all raw features"""
    return np.array([
        d["comp"],
        d["rational"],
        d["mem"],
        d["localmem"],
        d["coalesced"],
        d["atomic"],
        d["transfer"],
        d["wgsize"]
    ]).T


def static_features(d):
    """static features"""
    return np.array([
        d["comp"],
        d["mem"],
        d["localmem"],
        d["coalesced"],
    ]).T


def get_static_features(D):
    """static features from table"""
    return np.array([
        D["comp"].values,
        D["mem"].values,
        D["localmem"].values,
        D["coalesced"].values,
    ], dtype=float).T


def extended_static_features(d):
    """static features with branching"""
    return np.array([
        d["comp"],
        d["rational"],
        d["mem"],
        d["localmem"],
        d["coalesced"],
    ]).T


def getlabels(d):
    return d["oracle"]


class Metrics(object):
    def __init__(self, prefix, data, predicted, model=None):
        self._prefix = prefix
        self._data = data
        self._predicted = predicted
        self._model = model

    @property
    def prefix(self): return self._prefix

    @property
    def data(self): return self._data

    @property
    def predicted(self): return self._predicted

    @property
    def oracles(self):
        return [float(x) for x in self.data["speedup"]]

    @property
    def oracle(self):
        try:
            return self._oracle
        except AttributeError:
            assert(len(self.speedups) == len(self.oracles))
            self._oracle = self.speedup / labmath.geomean(self.oracles)
            return self._oracle

    @property
    def y_test(self):
        return self.data["oracle"]

    @property
    def accuracy(self):
        try:
            return self._accuracy
        except AttributeError:
            self._accuracy = accuracy_score(self.y_test, self.predicted)
            return self._accuracy

    @property
    def speedups(self):
        try:
            return self._speedups
        except AttributeError:
            speedups = []
            for d, p in zip(self.data.to_dict(orient="records"),
                            self.predicted):
                if d["oracle"] == p:
                    speedups.append(d["speedup"])
                else:
                    speedups.append(d["penalty"])
            self._speedups = np.array(speedups)
            return self._speedups

    @property
    def speedup(self):
        try:
            return self._speedup
        except AttributeError:
            self._speedup = labmath.geomean(self.speedups)
            return self._speedup

    @property
    def groups(self):
        try:
            return self._groups
        except AttributeError:
            self._groups = sorted(set(self.data["Group"]))
            return self._groups

    @property
    def n(self):
        return len(self.speedups)

    @property
    def model(self):
        return self._model

    def export_model(self, out_basename):
        try:
            outfile = fs.path(str(out_basename) + ".dot")
            tree.export_graphviz(self.model, out_file=outfile,
                                 max_depth=5, filled=True, rounded=True,
                                 class_names=["CPU", "GPU"],
                                 feature_names=["F1", "F2", "F3", "F4"])
            print("export model to '{}'".format(outfile))
        except Exception:
            pass

    header = ", ".join([
        "classifier",
        "accuracy",
        "speedup",
        "oracle"
    ])

    def __repr__(self):
        return ", ".join([
            self.prefix,
            "{:.2f}%".format(self.accuracy * 100),
            "{:.2f}".format(self.speedup),
            "{:.0f}%".format(self.oracle * 100)
        ])


def getgroups(data, getgroup):
    return sorted(list(set([getgroup(d) for d in
                            data.to_dict(orient="records")])))


def pairwise_groups_indices(data, getgroup):
    """
    """
    groups = getgroups(data, getgroup)

    group_indices = defaultdict(list)
    for i, d in enumerate(data.to_dict(orient="records")):
        group_indices[getgroup(d)].append(i)

    groupnames, pairs = [], []
    for j in range(len(groups)):
        for i in range(len(groups)):
            l, r = groups[j], groups[i]
            groupnames.append((l, r))
            li, ri = group_indices[l], group_indices[r]
            pairs.append((li, ri))
    return groupnames, pairs


def l1o_groups_indices(data, getgroup):
    """
    """
    groups = getgroups(data, getgroup)

    group_indices = defaultdict(list)
    for i, d in enumerate(data.to_dict(orient="records")):
        group_indices[getgroup(d)].append(i)

    groupnames, pairs = [], []
    for j in range(len(groups)):
        l = groups[j]
        groupnames.append((l, ", ".join([x for x in groups if x != l])))
        pairs.append(([item for sublist in
                       [group_indices[x] for x in groups if x != l]
                       for item in sublist],
                      group_indices[l]))
    return groupnames, pairs


def run_fold_indices(prefix, clf, data, train_index, test_index,
                     features=cgo13_features):
    X_train = features(data)[train_index]
    y_train = getlabels(data)[train_index]

    clf.fit(X_train, y_train)
    X_test = features(data)[test_index]

    predicted = clf.predict(X_test)

    predicted_data = data.ix[test_index]

    return Metrics(prefix, predicted_data, predicted, clf)


def run_test(prefix, clf, train, test, features=cgo13_features):
    X_train = features(train)
    y_train = getlabels(train)

    clf.fit(X_train, y_train)
    X_test = features(test)

    predicted = clf.predict(X_test)

    return Metrics(prefix, test, predicted, clf)


def run_xval(prefix, clf, data, cv, features=cgo13_features, seed=1):
    X = features(data)
    y = getlabels(data)

    predicted = cross_validation.cross_val_predict(clf, X, y, cv=cv)

    return Metrics(prefix, data, predicted, clf)


def model(seed=204):
    return DecisionTreeClassifier(
        random_state=seed, splitter="best", criterion="entropy")


def leave_one_benchmark_out(clf, get_features, D, benchmark):
    # Create data masks. For training we exclude all results from
    # the test benchmark.
    test_mask = D["benchmark"].str.contains(r"^" + benchmark)
    train_mask = ~test_mask

    # Create training and testing data:
    X_train = get_features(D[train_mask])
    y_train = getclass(D[train_mask])

    D_test = D[test_mask]
    X_test = get_features(D_test)
    y_test = getclass(D_test)

    # Train classifier:
    clf.fit(X_train, y_train)

    # Make predictions
    predicted = clf.predict(X_test)
    D_out = []
    for d, y, p in zip(D_test.to_dict('records'), y_test, predicted):
        d["p"] = p
        d["p_correct"] = 1 if y == p else 0
        D_out.append(d)

    # Return a list of dicts
    return D_out


def get_benchmark_names(data, prefix=None):
    if prefix:
        return sorted(set([
            re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
            for b in data["benchmark"] if b.startswith(prefix)
        ]))
    else:
        return sorted(set([
            re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
            for b in data["benchmark"]
        ]))


def xval_benchmarks(clf, data, **benchmark_name_opts):
    benchmark_names = get_benchmark_names(data, **benchmark_name_opts)
    return pd.DataFrame(
        flatten([leave_one_benchmark_out(clf, cgo13_features, data, b)
                 for b in benchmark_names]))


def classification(train, classifier="DecisionTree",
                   test=None, supplementary=None,
                   with_raw_features=False, only_raw_features=False,
                   group_by=None, samegroup_xval=False, l1o=False, **kwargs):
    if with_raw_features:
        getfeatures = cgo13_with_raw_features
    elif only_raw_features:
        getfeatures = raw_features
    else:
        getfeatures = cgo13_features

    seed = kwargs.get("seed", 0)

    # Get classifier
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=seed, criterion="entropy", splitter="best"),
        "NaiveBayes": GaussianNB(),
        "NearestNeighbour": KNeighborsClassifier(n_neighbors=1),
        "ZeroR": None,  # TODO:
    }
    lookup_table = {
        "DecisionTree": classifiers["DecisionTree"],
        "NaiveBayes": classifiers["NaiveBayes"],
        "NearestNeighbour": classifiers["NearestNeighbour"],
        "dt": classifiers["DecisionTree"],
        "nb": classifiers["NaiveBayes"],
        "nn": classifiers["NearestNeighbour"]
    }
    clf = lookup_table.get(classifier, None)
    if clf is None:
        raise Exception(
            "unkown classifier '{}'. Possible values: {{{}}}"
            .format(classifier, ",".join(sorted(lookup_table.keys()))))

    if test is not None:
        return run_test(classifier, clf, train, test, features=getfeatures)
    elif group_by:
        # Cross-validation over some grouping
        getgroup = {
            "suite": getsuite,
            "benchmark": getbenchmark,
        }.get(group_by, None)
        if group_by and not getgroup:
            raise Exception("Unkown group type '{}'".format(group_by))

        groups = sorted(getgroups(train, getgroup))

        if l1o:
            groupnames, folds = l1o_groups_indices(train, getgroup)
            results = [None] * len(groups)
        else:
            groupnames, folds = pairwise_groups_indices(train, getgroup)
            results = [[None] * len(groups) for x in range(len(groups))]

        for gpname, fold in zip(groupnames, folds):
            train_group, test_group = gpname
            train_index, test_index = fold

            # If samegroup_xval option is true, then cross-validate on
            # training data.
            if samegroup_xval and train_group == test_group:
                train2 = train.ix[train_index]
                metrics = classification(
                    train2, with_raw_features=with_raw_features,
                    only_raw_features=only_raw_features,
                    classifier=classifier, **kwargs)
            else:
                if supplementary is not None:
                    # If we have supplementary data, then copy data
                    # and append training.
                    train2 = train.ix[train_index]
                    train2 = train2.append(supplementary)

                    X_train = getfeatures(train2)
                    y_train = getlabels(train2)

                    clf.fit(X_train, y_train)

                    X_test = getfeatures(train)[test_index]

                    predicted = clf.predict(X_test)
                    predicted_data = train.ix[test_index]

                    metrics = Metrics(
                        classifier, predicted_data, predicted, clf)
                else:
                    metrics = run_fold_indices(classifier, clf, train,
                                               train_index, test_index,
                                               features=getfeatures)

            train_index = groups.index(train_group)
            if l1o:
                results[train_index] = metrics
            else:
                test_index = groups.index(test_group)
                results[train_index][test_index] = metrics

        return results
    else:
        # plain old cross-validation
        #
        # Get the number of folds to use. If "nfold=n", then perform
        # leave-one-out cross validation.
        nfolds = kwargs.get("nfolds", 10)
        if nfolds == "n":
            nfolds = len(train)
        else:
            nfolds = int(nfolds)

        folds = cross_validation.KFold(len(train), n_folds=nfolds,
                                       shuffle=True, random_state=seed)
        return run_xval(classifier, clf, train, folds, features=getfeatures)

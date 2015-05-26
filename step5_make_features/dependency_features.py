from __future__ import division

import os

import numpy as np
import pandas as pd

import lime_utils
import tree_utils

def dobj_metrics(t, fn_map, acc=None):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring('(A (dobj b) (dobj (C c) (D d)))')
    >>> width_fn = lambda t: len(t.leaves())
    >>> height_fn = lambda t: t.height()
    >>> dobj_metrics(t, {'width': width_fn, 'height': height_fn})
    {'width': [1, 2], 'height': [2, 3]}
    """
    if acc is None:
        acc = dict()

    if not tree_utils.is_tree(t):
        return acc

    is_dobj = t.label() == 'dobj'
    if is_dobj:
        for m in fn_map:
            acc[m] = acc.get(m, []) + [fn_map[m](t)]
    else:
        for s in t:
            dobj_metrics(s, fn_map, acc)

    return acc


def dep_feature_row(tr):
    dobj_fn_map = {
        'dobj_iota': tree_utils.iota,
        'dobj_hierarchy': tree_utils.hierarchy,
        'dobj_depth': lambda t: t.height(),
        'dobj_width': lambda t: len(t.leaves()),
        'dobj_shape': tree_utils.shape
    }

    metrics = dobj_metrics(tr, dobj_fn_map)

    return {'dep_' + m: np.nanmean(metrics[m]) for m in metrics}


_deps_file_filter = lambda f: f.endswith('deps.pkl')


def dirs_to_mat(patients_dir, controls_dir, src_filter=_deps_file_filter):
    return lime_utils.lime_nums_to_mat(patients_dir, controls_dir, src_filter, dep_feature_row)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

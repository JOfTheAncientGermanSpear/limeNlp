from __future__ import division

import numpy as np

import lime_utils
import tree_utils


def dep_metrics(t, fn_map, acc=None):
    """
    :param t
    :param fn_map
    :param acc
    >>> from nltk.tree import Tree
    >>> word = lambda w, i: 'word: %s, index: %s' % (w, i)
    >>> sent_tree = lambda i, t: Tree('id: %s' % i, [t])
    >>> sent_1 = sent_tree(1, Tree('dobj', [word('b', 1)])) 
    >>> sent_2 = sent_tree(2, Tree('dobj', 
    ...    [Tree('nsubj', [word('c', 1)]), word('d', 2)])) 
    >>> t = Tree('sentences', [sent_1, sent_2])
    >>> width_fn = lambda t: len(t.leaves())
    >>> height_fn = lambda t: t.height()
    >>> dep_metrics(t, {'width': width_fn, 'height': height_fn})
    {'dobj_width': [1, 2], 'dobj_height': [2, 3], 'nsubj_height': [2], 'nsubj_width': [1]}
    """
    if acc is None:
        acc = dict()

    if not tree_utils.is_tree(t):
        return acc

    label = t.label()
    is_word_label = "word: " in label
    is_tag = not (lime_utils.is_sentence_label(label) or is_word_label)
    if is_tag:
        for m in fn_map:
            k = label + "_" + m
            acc[k] = acc.get(k, []) + [fn_map[m](t)]

    for s in t:
        dep_metrics(s, fn_map, acc)

    return acc


def dep_feature_row(tr):
    fn_map = {
        'iota': tree_utils.iota,
        'iota_denom': tree_utils.iota_denom,
        'hierarchy': tree_utils.hierarchy,
        'depth': lambda t: t.height(),
        'width': lambda t: len(t.leaves()),
        'shape': tree_utils.shape
    }

    metrics = dep_metrics(tr, fn_map)

    return {'dep_' + m: np.nanmean(metrics[m]) for m in metrics}


_deps_file_filter = lambda f: f.endswith('deps.pkl')


def dirs_to_mat(patients_dir, controls_dir, src_filter=_deps_file_filter):
    return lime_utils.lime_nums_to_mat(patients_dir, controls_dir, src_filter, dep_feature_row)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

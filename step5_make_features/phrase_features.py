from __future__ import division

import os

import nltk
import numpy as np
import pandas as pd

import lime_utils
import tree_utils


def is_sentence_label(l):
    """
    >>> is_sentence_label('id: 0')
    True
    >>> is_sentence_label('id: 37')
    True
    >>> is_sentence_label('sentences')
    True
    >>> is_sentence_label('ab')
    False
    """
    l_lower = l.lower()
    return l_lower == "sentences" or 'id: ' in l_lower


def avg_vals_fn(d):
    """
    >>> import numpy as np
    >>> d = {'a': [1, 2, 3], 'b': [4, 5], 'c': [3, np.nan, 5]}
    >>> fn = avg_vals_fn(d)
    >>> fn('a')
    2.0
    >>> fn('b')
    4.5
    >>> fn('c')
    4.0
    """
    def fn(k):
        vals = filter(lambda v: not np.any(np.isnan(v)), d[k])
        if len(vals) > 0:
            return np.average(vals, axis=0)

    return fn


def avg_metric_by_label(t, metric_fn, label_attachment):
    """
    >>> from nltk.tree import Tree
    >>> t_0 = Tree.fromstring("(A (B b))")
    >>> t_1 = Tree.fromstring("(A a)")
    >>> fn = lambda t: t.height()
    >>> t = Tree("sentences", [Tree("id: 0", [t_0]), Tree("id: 1", [t_1])])
    >>> avg_heights = avg_metric_by_label(t, fn, '_avg_height')
    >>> sorted(avg_heights.keys())
    ['A_avg_height', 'B_avg_height']
    >>> [avg_heights[l] for l in sorted(avg_heights.keys())]
    [2.5, 2.0]
    """
    is_not_sentence_label = lambda t: not is_sentence_label(t.label())
    metric_by_label = nltk.ConditionalFreqDist((s.label(), metric_fn(s))
                                               for s in t.subtrees(filter=is_not_sentence_label))

    avg_metric = avg_vals_fn(metric_by_label)

    def set_metric(fd, label):
        a = avg_metric(label)
        if a is not None:
            fd[label + label_attachment] = a
        return fd

    return reduce(set_metric, metric_by_label, nltk.FreqDist())


def phrase_shapes(t):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> shapes = phrase_shapes(t)
    >>> sorted(shapes.keys())
    ['DT_avg_depth', 'DT_avg_width', 'NN_avg_depth', 'NN_avg_width', 'NP_avg_depth', 'NP_avg_width', 'S_avg_depth', 'S_avg_width', 'VBD_avg_depth', 'VBD_avg_width', 'VP_avg_depth', 'VP_avg_width']
    >>> [shapes[k] for k in sorted(shapes.keys())]
    [1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0]
    >>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
    >>> shapes = phrase_shapes(t)
    >>> sorted(shapes.keys())
    ['DT_avg_depth', 'DT_avg_width', 'NN_avg_depth', 'NN_avg_width', 'NP_avg_depth', 'NP_avg_width', 'S_avg_depth', 'S_avg_width', 'VBD_avg_depth', 'VBD_avg_width', 'VP_avg_depth', 'VP_avg_width']
    >>> [shapes[k] for k in sorted(shapes.keys())]
    [1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0]
    """
    avg_shape_by_label = avg_metric_by_label(t, tree_utils.shape, '')

    def set_w_d(fd, label):
        s = avg_shape_by_label.get(label, None)
        if s is not None:
            fd[label+'_avg_width'] = s[0]
            fd[label+'_avg_depth'] = s[1]
        return fd

    return reduce(set_w_d, avg_shape_by_label, nltk.FreqDist())


def tag_counts(t):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> p = tag_counts(t)
    >>> sorted(p.keys())
    ['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
    >>> [p[k] for k in sorted(p.keys())]
    [2, 2, 2, 1, 1, 1]
    >>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
    >>> p = tag_counts(t)
    >>> sorted(p.keys())
    ['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
    >>> [p[k] for k in sorted(p.keys())]
    [4, 4, 4, 2, 2, 2]
    """
    return nltk.FreqDist(s.label() for s in t.subtrees() if not is_sentence_label(s.label()))


def phrase_sentence_cover(t, coeff=1.0, covers=None):
    """
    >>> from nltk.tree import Tree
    >>> s = "(A (B (C c) (D d)) (E e))"
    >>> t = Tree.fromstring(s)
    >>> c = phrase_sentence_cover(t)
    >>> sorted(c.keys())
    ['A', 'B', 'C', 'D', 'E']
    >>> [c[k] for k in sorted(c.keys())]
    [1.0, 0.5, 0.25, 0.25, 0.5]
    >>> s_small = "(A (B (C c) (D d)))"
    >>> t_small = Tree.fromstring(s_small)
    >>> t = Tree("sentences", [Tree("id: 0", [t]), Tree("id: 1", [t_small])])
    >>> c = phrase_sentence_cover(t)
    >>> sorted(c.keys())
    ['A', 'B', 'C', 'D', 'E']
    >>> [c[k] for k in sorted(c.keys())]
    [1.0, 0.75, 0.375, 0.375, 0.5]
    """
    if covers is None:
        covers = dict()

    if not tree_utils.is_tree(t):
        return covers

    label = t.label()

    if is_sentence_label(label):
        covers_per_sent = map(lambda s: phrase_sentence_cover(s, coeff, dict()), t)
        return tree_utils.avg_dicts(covers_per_sent)

    covers[label] = covers.get(label, 0) + coeff

    num_children = len(t)

    for c in t:
        phrase_sentence_cover(c, coeff / num_children, covers)

    return covers


def number_clauses(p):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (SBAR (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> p = number_clauses(t)
    >>> p
    2
    >>> tags = tag_counts(t)
    >>> number_clauses(tags)
    2
    """
    if tree_utils.is_tree(p):
        p = tag_counts(p)

    clause_level = {'S', 'SBAR', 'SBARQ', 'SINV', 'SQ'}

    return sum([p[c] for c in p if c in clause_level])


def tag_ratios(p):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> p = tag_ratios(t)
    >>> sorted(p.keys())
    ['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
    >>> vals = [p[k] for k in sorted(p.keys())]
    >>> vals == [2.0/9, 2.0/9, 2.0/9, 1.0/9, 1.0/9, 1.0/9] 
    True
    """
    if tree_utils.is_tree(p):
        p = tag_counts(p)
    
    ratios = nltk.FreqDist()
    for l in p:
        ratios[l] = p.freq(l)
    
    return ratios


def phrase_hierarchies(t):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> hiers = phrase_hierarchies(t)
    >>> sorted(hiers.keys())
    ['NP_avg_hier', 'S_avg_hier', 'VP_avg_hier']
    >>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
    >>> hiers = phrase_hierarchies(t)
    >>> sorted(hiers.keys())
    ['NP_avg_hier', 'S_avg_hier', 'VP_avg_hier']
    """

    return avg_metric_by_label(t, tree_utils.hierarchy, '_avg_hier')


def phrase_iotas(t):
    """
    >>> from nltk.tree import Tree
    >>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
    >>> t = Tree.fromstring(s)
    >>> iotas = phrase_iotas(t)
    >>> sorted(iotas.keys())
    ['DT_avg_iota', 'NN_avg_iota', 'NP_avg_iota', 'S_avg_iota', 'VBD_avg_iota', 'VP_avg_iota']
    >>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
    >>> iotas = phrase_iotas(t)
    >>> sorted(iotas.keys())
    ['DT_avg_iota', 'NN_avg_iota', 'NP_avg_iota', 'S_avg_iota', 'VBD_avg_iota', 'VP_avg_iota']
    """

    return avg_metric_by_label(t, tree_utils.iota, '_avg_iota')


def phrase_dists(t):
    """
    >>> from nltk.tree import Tree
    >>> s = "(A (B b) (C c))"
    >>> t = Tree.fromstring(s)
    >>> dists = phrase_dists(t)
    >>> sorted(dists.keys())
    ['A_B_avg_dist', 'A_C_avg_dist', 'B_C_avg_dist']
    >>> [dists[k] for k in sorted(dists.keys())]
    [1.0, 1.0, 2.0]
    >>> t2 = Tree.fromstring("(A (D (B b)))")
    >>> t_sents = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t2])])
    >>> dists = phrase_dists(t_sents)
    >>> sorted(dists.keys())
    ['A_B_avg_dist', 'A_C_avg_dist', 'A_D_avg_dist', 'B_C_avg_dist', 'B_D_avg_dist']
    >>> [dists[k] for k in sorted(dists.keys())]
    [1.5, 1.0, 1.0, 2.0, 1.0]
    >>> s = "(A (B b) (A a))"
    >>> t = Tree.fromstring(s)
    >>> dists = phrase_dists(t)
    >>> sorted(dists.keys())
    ['A_A_avg_dist', 'A_B_avg_dist']
    >>> [dists[k] for k in sorted(dists.keys())]
    [1.0, 1.5]
    """

    if t.label().lower() == "sentences":
        sentence_roots = (c[0] for c in t.subtrees() if 'id: ' in c.label())
    else:
        sentence_roots = [t]

    distances_by_connection = dict()

    for s in sentence_roots:
        label = lambda p: s[p].label()
        is_tree_at_pos = lambda p: tree_utils.is_tree(s[p])
        posits = filter(is_tree_at_pos, s.treepositions())
        for (i, p) in enumerate(posits):
            for p2 in posits[i + 1:]:
                (start, stop) = sorted([label(p), label(p2)])
                connection = start + '_' + stop
                dists = distances_by_connection.get(connection, [])
                dists.append(tree_utils.dist(p, p2))
                distances_by_connection[connection] = dists

    def avg(acc, k):
        dists = distances_by_connection[k]
        acc[k + '_avg_dist'] = np.average(dists)
        return acc

    return reduce(avg, distances_by_connection, dict())


def phrase_yngve_depths(t):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring('(A (B b) (C c (B b)))')
    >>> d = phrase_yngve_depths(t)
    >>> expected = {'A_avg_yngve_depth': 0, 'B_avg_yngve_depth': .5, 'C_avg_yngve_depth': 0}
    >>> assert(expected == d)
    """

    is_not_sentence_label = lambda tree: not is_sentence_label(tree.label())

    concat = dict()
    for sub_tree in tree_utils.below_condition(t, is_not_sentence_label):
        sub_tree_yngve = tree_utils.yngve_depth(sub_tree)
        for label in sub_tree_yngve:
            concat[label] = concat.get(label, []) + sub_tree_yngve[label]

    return {l+'_avg_yngve_depth': np.average(concat[l]) for l in concat}


def number_sentences(t):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree("sentences", [Tree("id: 0", ['a']), Tree("id: 1", ['b'])])
    >>> number_sentences(t)
    2
    """
    return len(t)


def phrase_feature_row(t):
    def labeled_series(fd, l=None):
        return pd.Series({(k + '_' + l) if l else k: fd[k] for k in fd})

    counts = labeled_series(tag_counts(t), 'count')
    ratios = labeled_series(tag_ratios(t), 'ratio')
    clause_count = labeled_series({'clause_count': number_clauses(counts)})
    sentence_count = labeled_series({'sentence_count': number_sentences(t)})
    shapes = pd.Series(phrase_shapes(t))
    hierarchy = pd.Series(phrase_hierarchies(t))
    iota = pd.Series(phrase_iotas(t))
    dists = pd.Series(phrase_dists(t))
    yngve_depths = pd.Series(phrase_yngve_depths(t))

    return pd.concat([counts, ratios, clause_count, sentence_count,
                      shapes, hierarchy, iota, dists, yngve_depths])


_phrase_file_filter = lambda f: 'phrase' in f and f.endswith('.pkl')


def dirs_to_mat(patients_src_dir, controls_src_dir, src_filter=_phrase_file_filter):

    return lime_utils.lime_nums_to_mat(patients_src_dir, controls_src_dir, src_filter, phrase_feature_row)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

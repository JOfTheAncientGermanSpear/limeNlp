from __future__ import division

import os

import nltk
from nltk.tree import Tree
import numpy as np
import pickle
from scipy import stats


def is_tree(t):
    return isinstance(t, nltk.Tree)


def dist(pos1, pos2):
    """
    >>> dist((0, 0), (0, 1))
    2
    >>> dist((1,), (1, 1))
    1
    >>> dist((), (1, 1))
    2
    >>> dist((1,2), (0, 0, 3))
    5
    >>> dist((1,2), (1, 0, 3))
    3
    >>> dist((0, 1), (1, 1))
    4
    >>> dist((0, 1), (0, 1))
    0
    """

    pos1_level = len(pos1)
    pos2_level = len(pos2)

    (higher, lower) = (pos1, pos2) if pos1_level <= pos2_level \
            else (pos2, pos1)

    diff_ix = (i for (i, h) in enumerate(higher) if not h == lower[i])
    num_shared = next(diff_ix, len(higher))

    return len(lower) + len(higher) - 2 * num_shared


def shape(t):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring("(S (NP (NNP Bob)) (VP sleeps))")
    >>> shape(t)
    (4, 2)
    """
    return (t.height(), len(t.leaves()))


def order(t, is_root = True):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring("(A (B b))")
    >>> order(t)
    1
    >>> order(t[0], False) #B subtree
    2
    """
    num_parents = 0 if is_root else 1

    if not is_tree(t):
        return num_parents

    num_children = len(t)
    return num_children + num_parents
    

def orders(t, is_root = True):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring("(A a)")
    >>> orders(t)
    array([1, 1])
    >>> t = Tree.fromstring("(A (B b))")
    >>> orders(t)
    array([1, 2, 1])
    """
    
    o = np.array([order(t, is_root)])

    if not is_tree(t):
        return o

    def app_child_order(acc, c):
        c_orders = orders(c, is_root = False)
        return np.concatenate([acc, c_orders])

    return reduce(app_child_order, t, o)


def hierarchy(t):
    """
    >>> from nltk.tree import Tree
    >>> o_1 = Tree.fromstring("(1)")
    >>> o_11 = Tree(11, [o_1] * 10)
    >>> r = Tree(10, [o_11] * 10)
    >>> actual = hierarchy(r)
    >>> from scipy import stats
    >>> import numpy as np
    >>> degrees = [1, 10, 11]
    >>> cnts = [100, 1, 10]
    >>> s = stats.linregress(np.log10(degrees), np.log10(cnts))[0]
    >>> s == actual
    True
    """

    os = orders(t)
    o_counts = dict()
    for o in os:
        o_counts[o] = o_counts.get(o, 0) + 1
    
    degrees = sorted(o_counts.keys())
    cnts = [o_counts[d] for d in degrees]

    slope = stats.linregress(np.log10(degrees), np.log10(cnts))[0]
    return slope


def iota(t, is_root = True):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring("(A (B b))")
    >>> iota(t) #1 + 2*2 + 1
    6
    >>> t = Tree.fromstring("(A (B (C (D d) (D d))))")
    >>> (a, b, c, d, d_l) = (1, 2*2, 2*3, 2*2*2, 2*1) 
    >>> iota(t) == a + b + c + d + d_l
    True
    """

    os = orders(t, is_root)

    def iota_sum(acc, o):
        w = o if o < 2 else o * 2
        return acc + w

    return reduce(iota_sum, os, 0)


def avg_dicts(ds):
    """
    >>> d_a = {'a': 3, 'b': 4}
    >>> d_b = {'a': 4, 'b': 5, 'c': 3}
    >>> d_avg = avg_dicts([d_a, d_b]) 
    >>> [d_avg[k] for k in sorted(d_avg.keys())]
    [3.5, 4.5, 3.0]
    """
    key_counts = dict()

    def running_avg(avg, d):
        for k in d:
            prev_count = key_counts.get(k, 0)
            curr_count = prev_count + 1
            key_counts[k] = curr_count

            prev_avg = avg.get(k, 0)
            avg[k] = (prev_avg * prev_count + d[k])/curr_count
        return avg

    return reduce(running_avg, ds, dict())


def merge_trees(trees, label, replace_roots=False):
    """
    >>> from nltk.tree import Tree
    >>> a1 = Tree("a", [Tree('B', ['b']), Tree('C', ['c'])])
    >>> a2 = Tree("a", [Tree('D', ['d']), Tree('E', ['e'])])
    >>> t = merge_trees([a1, a2], label='r', replace_roots = True)
    >>> str(t.pprint())
    '(r (B b) (C c) (D d) (E e))'
    >>> t = merge_trees([a1, a2], label='r', replace_roots = False)
    >>> str(t.pprint())
    '(r (a (B b) (C c)) (a (D d) (E e)))'
    >>> t = merge_trees([a1, Tree('b', ['e'])], label='r', replace_roots = True)
    >>> str(t.pprint())
    '(r (B b) (C c) e)'
    """

    def all_children():
        def append_children(acc, t):
            return acc + [c for c in t]
        return reduce(append_children, trees, [])

    children = all_children() if replace_roots else trees
    return Tree(label, children)


def yngve_depth(tree, include_leaves=False, _current_depth=0, _ret=None):
    """
    >>> from nltk.tree import Tree
    >>> t = Tree.fromstring('(A (B b) (C c (B b)))')
    >>> d = yngve_depth(t, include_leaves=True)
    >>> expected = {'A': [0], 'B': [1, 0], 'C': [0], 'b': [1, 0], 'c': [1]}
    >>> assert(d == expected)
    """
    if _ret is None:
        _ret = dict()

    is_leaf = lambda t: not is_tree(t)

    def append_label(l):
        _ret[l] = _ret[l] + [_current_depth] if l in _ret else [_current_depth]

    if is_leaf(tree):
        append_label(tree)
    else:
        append_label(tree.label())
        num_children = len(tree)
        for i, child in enumerate(tree):
            num_younger_sisters = num_children - 1 - i
            keep_going = is_tree(child) or (is_leaf(child) and include_leaves)
            if keep_going:
                yngve_depth(child, include_leaves, _current_depth + num_younger_sisters, _ret)

    return _ret


def below_condition(tree, condition_fn):
    """
    >>> from nltk.tree import Tree
    >>> sents = Tree.fromstring('(sents ("id: 0" (A a))("id: 1" (B b)))')
    >>> not_sent = lambda s: 'sents' not in s.label() and 'id:' not in s.label()
    >>> trees = below_condition(sents, not_sent)
    >>> trees
    [Tree('A', ['a']), Tree('B', ['b'])]
    """
    if not is_tree(tree):
        return []

    if condition_fn(tree):
        return [tree]
    else:
        ret = []
        for c in tree:
            ret = ret + below_condition(c, condition_fn)
        return ret


def load_tree(tree_path):
    with open(tree_path, 'rb') as f:
        t = pickle.load(f)
    return t


if __name__ == "__main__":
    import doctest
    doctest.testmod()
